import re
import os
import time
os.environ["GRADIO_TEMP_DIR"] = "./gradio_tmp"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_USE_V1"] = "0"
import torch
import asyncio
import gradio as gr
from collections import defaultdict
from transformers import AutoProcessor, AutoModel, GenerationConfig
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid
from remote_code.media import extract_media
from remote_code.tokenizer_utils import tokenize_conversation
from remote_code.mm_utils import process_images
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from gradio.themes.utils.colors import Color as ColorTemplate
from nvidia import Nvidia
from gradio.themes.utils import colors, fonts, sizes

question_mapping = {
    "images/depu.mp4": "This video features a Texas Hold'em poker match between two players: Thomas Mühlöcker (wearing sunglasses and a navy-blue shirt) and Fedor Holz (wearing a black jacket). Thomas Mühlöcker holds the Ace of Clubs and the King of Hearts, while Fedor Holz's cards remain unknown. The community cards are as follows: Flop: 7 of Clubs, Jack of Hearts, 2 of Hearts; Turn: 8 of Diamonds; River: King of Diamonds (visible in the video). Based on the betting patterns observed, assume you are Thomas Mühlöcker and know that your best possible hand is a pair of Kings. After analyzing the video, infer Fedor Holz's hand.\nA. Club King, Club Jack\nB. Heart 3, Heart 10\nC. Spade King, Club King\nD. Diamond Ace, Spade Ace",
    # "images/football-game.mp4": "You are shown a 30-minute video segment from a football game, covering the extra time (90th to 120th minute). The score remains 2–2 throughout this period, and no goals are scored. Based on players’ physical condition, tactical behavior, emotional state, and overall match performance observed during extra time, which team is more likely to win the upcoming penalty shootout? What is the most likely final result?\nA. The Netherlands win the penalty shootout 4–2, thanks to their momentum and psychological advantage from coming back in regulation time.\nB. Argentina win the penalty shootout 4–3, with their goalkeeper playing a decisive role and key players maintaining composure.\nC. The Netherlands win the shootout 5–4 after a flawless penalty performance and two missed shots by Argentina.\nD. Argentina score a goal in the final minutes of extra time and win 3–2 without going to penalties.",
    # "images/moving-cup.mp4": "There are three boxes, and a purple ball is initially placed in the middle box. Then the positions of the boxes are swapped. Please analyze step by step the movement trajectory of the box containing the ball through the video and determine the final position of the ball.\nA. left\nB. middle\nC. right\nD. none of the above",
    "images/taboo.mp4": "This a video of two people playing the game Taboo on the topic of 'machine learning.' One person describes a word, and the other person guesses it. Your task is to analyze the video and choose the most likely word being described from the following options:\nA. Logistic Regression - Pruning - Attention Mechanism\nB. Support Vector Machine - Quantization - Transformer\nC. Naive Bayes - Pruning - Gradient Boosting\nD. Neural Network - Quantization - Transformer",
    # "images/starcraft-2.mp4": "You are presented with the first 20 minutes of a StarCraft II game. Based on the observed strategies, unit compositions, expansions, and tactical maneuvers during this period, what are the likely strategic intentions of both players, their next possible actions, and who is more likely to win the match?\nA. Zest executes a successful Stargate-based air harassment strategy, gaining map control and eventually overwhelming Reynor with superior air units.\nB. Reynor adapts to Zest's early aggression, transitions into a Roach-Ravager composition, and secures victory through sustained ground assaults.\nC. Zest's early proxy strategy catches Reynor off-guard, leading to a quick win before Reynor can stabilize his economy.\nD. Both players engage in a prolonged macro game, but Zest's superior late-game unit composition allows him to outmaneuver Reynor and claim victory.",
    # "images/avenger-endgame.mp4": "What narrative intention is most strongly supported by the juxtaposition of Tony Stark's focused activity in the spaceship and the cemetery scene showing his gravestone?",
    # "images/cat-news.mp4": "What is the main purpose of the televised segments described in the video?",
    # "images/marvel-black-panther.mp4": "In 'Captain America: Civil War', why does T'Challa initially pursue Bucky Barnes?",
    # "images/BL5G6C_6gYY.mp4": "Which team is most likely associated with the 'BOSS' sponsorship, and what evidence supports this affiliation?"
}

class StableLMBot:
    def __init__(self):
        model_path = "/mnt/disk01/bhzhang/dataset/LongVILA-R1-7B"
        
        # 由于只有 1 张 H800，这里需要先加载 vllm 引擎（vllm 不允许加载时显存被占用）
        # 先初始化 vLLM 引擎（在加载主模型之前）
        self.inference_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=os.path.join(model_path, "llm"),
                gpu_memory_utilization=0.5,  # 为主模型留出空间
                enable_prompt_embeds=True,
                seed=42,
                disable_log_stats=True,
            )
        )
        
        # 然后加载主模型（vision encoder）
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto", llm_only_need_embed=True)
        self.model.eval()

        sampling_params_deep_think = {
            "n": 1,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 1024,
            "stop": None,
            "detokenize": True,
            "seed": 42,
        }
        self.sampling_params_deep_think = sampling_params_deep_think

        sampling_params_plain = {
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 1024,
            "stop": None,
            "detokenize": True,
            "seed": 42,
        }
        self.sampling_params_plain = sampling_params_plain

        self.num_video_frame = 256
        self.model.config.num_video_frames = self.num_video_frame

        self.generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=False,
            temperature=1,  # HACK
            num_return_sequences=1,
        )
        self.system_prompt_w_thinking = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: "
        self.cache_embeds = {}
        for video in question_mapping.keys():
            video_name = os.path.basename(os.path.basename(video))
            cache_path = video.replace(".mp4", ".pt")
            if os.path.exists(cache_path):
                print("Loading cached video embeddings {}".format(video_name))
                self.cache_embeds[video_name] = torch.load(cache_path, weights_only=False)

    async def generate_response(self, video_path, user_input, use_thinking):
        start = time.time()
        print("Generating response...")

        if not video_path:
            yield "Please upload a video or select an example before asking a question!"
            return
        if not user_input.strip():
            yield "Your question is empty!"
            return
        
        if use_thinking:
            prompt = self.system_prompt_w_thinking + user_input
        else:
            prompt = user_input

        video_name = os.path.basename(video_path)
        start_total = time.time()  # 记录总开始时间
        use_cache = video_name in self.cache_embeds
        conversation = [{"from": "human", "value": [prompt, {"path": video_path}]}]
        media = extract_media(conversation, self.model.config, draft=use_cache)
        if "<vila/video>" in conversation[0]["value"]:
            conversation[0]["value"] = "<vila/video>" + conversation[0]["value"].replace("<vila/video>", "")

        print("Tokenize Conversation...")
        input_ids = tokenize_conversation(conversation, self.model.tokenizer, add_generation_prompt=True).unsqueeze(0).cuda()

        media_config = defaultdict(dict, {})
        if use_cache:
            media = self.cache_embeds[video_name]
            media_config["use_cache"] = True
            print("Cache loaded...")
        else:
            # Process media
            media["video"] = [
                process_images(images, self.model.vision_tower.image_processor, self.model.config).half()
                for images in media["video"]
            ]
            print("Processed images")

        inputs_embeds, _, _ = self.model._embed(input_ids, media, media_config, None, None)
        print("Finished  embeddings...")

        request_id = random_uuid()
        results_generator = self.inference_engine.generate(
            {"prompt_embeds": inputs_embeds.squeeze(0)},
            request_id=request_id,
            sampling_params=SamplingParams(**self.sampling_params_deep_think) if use_thinking else SamplingParams(**self.sampling_params_plain),
        )

        async for request_output in results_generator:
            for output in request_output.outputs:
                new_text = output.text
                print(new_text)
                yield new_text

        end = time.time()
        print(f"Generation completed in {end - start:.2f} seconds.")

        # embeds_list = inputs_embeds.squeeze(0).cpu().tolist()
        # print(f"Embed list format check: type={type(embeds_list)}, len={len(embeds_list)}, first_elem_type={type(embeds_list[0])}")

bot = StableLMBot()

def set_example_video(example_path):
    question = question_mapping.get(example_path, "Describe the video.")
    return gr.update(value=example_path), gr.update(value=question)

#async def inference(video_path, user_input, use_thinking):
#    async for chunk in bot.generate_response(video_path, user_input, use_thinking):
#        yield chunk

async def inference(video_path, user_input, use_thinking):
    response = ""
    async for chunk in bot.generate_response(video_path, user_input, use_thinking):
        response += chunk
        yield [(user_input, response)]  # format for gr.Chatbot

async def inference(video_path, user_input, use_thinking):
    response = ""
    async for delta in bot.generate_response(video_path, user_input, use_thinking):
        #response += delta
        yield delta  # 每次刷新整个 textbox

nv_green = ColorTemplate(
    name="nv_green",
    c50="#f2f9f3",  # Light shade of nv green
    c100="#e5f3eb",  # Lighter shade of nv green
    c200="#c7e6d9",  # Mid-tone nv green
    c300="#76b900",  # nv green
    c400="#48873a",  # Darker shade of nv green
    c500="#30692c",  # Dark shade of nv green
    c600="#245121",  # Very dark shade of nv green
    c700="#1b3a17",  # Very dark shade of nv green
    c800="#12280e",  # Very dark shade of nv green
    c900="#0a1909",  # Very dark shade of nv green
    c950="#081407",  # Very dark shade of nv green
)

# 创建 Gradio 界面
with gr.Blocks(
    theme=Nvidia(
        primary_hue=nv_green,
        secondary_hue=nv_green,
        neutral_hue=colors.gray,
    ),
    title="LongVILA-R1 Playground",

    js="async () => {" + open("./app_script.js", "r", encoding="utf-8").read() + "}",
    css="./app_style.css",) as demo:
    # 顶部标题
    with gr.Column(scale=1, min_width=400):
        gr.Image("images/logo.png", show_label=False, elem_id="logo")
        gr.Markdown(
            """
            <h1 style="text-align: center; font-size: 35px; margin-top: 1px;">
                Scaling RL to Long Videos
            </h1>
            """
        )
    gr.Markdown(
        """
        <div style="text-align: center;">
            <strong style="font-size: 25px;">LongVILA-R1-7B</strong><br><br>
            <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
                <a href='https://github.com/NVlabs/Long-RL'>
                    <img src='https://img.shields.io/badge/GitHub-Long%20RL-blue' alt='GitHub'>
                </a>
                <a href='https://huggingface.co'>
                    <img src='https://img.shields.io/badge/HF%20Model-LongVILA%20R1-bron' alt='GitHub'>
                </a>
                <a href='https://arxiv.org/pdf/2507.07966'>
                    <img src='https://img.shields.io/badge/ArXiv-Paper-red' alt='ArXiv'>
                </a>
                <a href='https://www.youtube.com/watch?v=ykbblK2jiEg'>
                    <img src='https://img.shields.io/badge/YouTube-Intro-yellow' alt='YouTube'>
                </a>
            </div>
        </div>
        <strong> </strong>
        <p>
            <strong>LongVILA-R1-7B</strong> demonstrates strong performance in long video reasoning, achieving <strong>70.7%</strong> on VideoMME (w/ sub.) and surpassing Gemini-1.5-Pro across diverse reasoning tasks.<br> 
            <strong>Long-RL</strong> is a codebase that accelerates long video RL training by up to <strong>2.1×</strong> through its MR-SP system. It supports RL training on image, video, and omni inputs across VILA, Qwen/Qwen-VL, and diffusion models.
        </p>
        <p>
            <strong>LongVILA-R1-7B</strong> allows users to choose whether to enable deep reasoning during answering (by clicking the "<strong>Deep Thinking</strong>" button).<br>
             <strong>LongVILA-R1-7B</strong> supports both <u>multiple-choice</u> questions (as the first 5 examples) and <u>open-ended</u> questions (as the last 3 examples).
        </p>
        """
    )

    # 上传和示例部分
    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            # 视频上传
            input_video_path = gr.Video(label="Input Video") #,elem_attributes={"data-upload-text": "Drag and drop a video here or click to upload"})

            gr.Markdown(
                """
                <h1 style="text-align: left; font-size: 15px; margin-top: 1px;">
                    Examples:
                </h1>
                """
            )
            with gr.Row():
                example_button_1 = gr.Button("Texas Hold'em", elem_classes="example-button")
                example_button_2 = gr.Button("Football Game", elem_classes="example-button")
                example_button_3 = gr.Button("Moving Cup", elem_classes="example-button")
                example_button_4 = gr.Button("Taboo", elem_classes="example-button")
                example_button_5 = gr.Button("StarCraft II", elem_classes="example-button")
                example_button_6 = gr.Button("Avenger Endgame", elem_classes="example-button")
                example_button_7 = gr.Button("Cat News", elem_classes="example-button")
                example_button_8 = gr.Button("Black Panther", elem_classes="example-button")
            # 用户问题输入框
            input_text = gr.Textbox(
                show_label=True,
                placeholder="Enter your question here...",
                label="User Question",
                lines=5.2,
            )
            # 切换 "thinking" 能力的按钮
            use_thinking_toggle = gr.Checkbox(
                label="Deep Thinking",
                value=True
            )
            # 按钮
            run_button = gr.Button("🏃‍♂️ Run", elem_classes="custom-button")
            output_result = gr.Textbox(
                label="Result",
                show_label=True,
                lines=14,  # 调整文本框高度
                interactive=False,
                autoscroll = True
            )

    def load_example_1():
        return set_example_video("images/depu.mp4")

    def load_example_2():
        return set_example_video("images/football-game.mp4")

    def load_example_3():
        return set_example_video("images/moving-cup.mp4")

    def load_example_4():
        return set_example_video("images/taboo.mp4")

    def load_example_5():
        return set_example_video("images/starcraft-2.mp4")

    def load_example_6():
        return set_example_video("images/avenger-endgame.mp4")

    def load_example_7():
        return set_example_video("images/cat-news.mp4")

    def load_example_8():
        return set_example_video("images/marvel-black-panther.mp4")

    example_button_1.click(
        fn=load_example_1,
        inputs=[],
        outputs=[input_video_path, input_text],  # 将示例视频路径加载到视频组件
    )

    example_button_2.click(
        fn=load_example_2,
        inputs=[],
        outputs=[input_video_path, input_text],  # 将示例视频路径加载到视频组件
    )
    example_button_3.click(
        fn=load_example_3,
        inputs=[],
        outputs=[input_video_path, input_text],  # 将示例视频路径加载到视频组件
    )
    example_button_4.click(
        fn=load_example_4,
        inputs=[],
        outputs=[input_video_path, input_text],  # 将示例视频路径加载到视频组件
    )
    example_button_5.click(
        fn=load_example_5,
        inputs=[],
        outputs=[input_video_path, input_text],  # 将示例视频路径加载到视频组件
    )
    example_button_6.click(
        fn=load_example_6,
        inputs=[],
        outputs=[input_video_path, input_text],  # 将示例视频路径加载到视频组件
    )
    example_button_7.click(
        fn=load_example_7,
        inputs=[],
        outputs=[input_video_path, input_text],  # 将示例视频路径加载到视频组件
    )
    example_button_8.click(
        fn=load_example_8,
        inputs=[],
        outputs=[input_video_path, input_text],  # 将示例视频路径加载到视频组件
    )

    # Run 按钮点击事件
    run_button.click(
        fn=inference,
        inputs=[input_video_path, input_text, use_thinking_toggle],
        outputs=output_result,
    )

# 启动 Gradio 界面
# Gradio 会从 7800 开始尝试，如果占用就尝试下一个端口
import socket

def find_free_port(start_port=7800, max_port=7900):
    """找到一个可用的端口"""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError(f"Cannot find empty port in range: {start_port}-{max_port}")

free_port = find_free_port()
print(f"Starting Gradio on port {free_port}")
demo.launch(share=True, server_port=free_port, server_name="0.0.0.0")

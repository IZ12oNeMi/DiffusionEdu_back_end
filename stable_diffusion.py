import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import time
import os
from transformers import pipeline

from huggingface_hub import login

# -------------------------------------------加载全局项--------------------------------------------
login(
    token="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"
)  # 可以换成huggingface自己账号的token，或者在终端进行登陆验证
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
# 导入预训练模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
# 使用DDIM加速图像生成过程
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
# 优化内存管理
pipe.enable_attention_slicing()

# 设定标签
tags = {
    "1": ", vibrant colors",
    "2": ", historical style",
    "3": ", sunny day",
    "4": ", scientific style",
    "5": ", high detail",
    "6": ", illustration",
    "7": ", realistic style",
    "8": ", cartoon style",
    "9": ", watercolor style",
    "10": ", ultra-realistic",
    "11": ", labeled components",
    "12": ", night scene",
    "13": ", educational diagram",
    "14": ", geological feature",
}


# -------------------------------------------方法定义--------------------------------------------
def translate_prompt(prompt):
    """
    中译英，并加入场景下通用的提示词

    Args:
        prompt (str): 输入的prompt.

    Returns:
        str: 转化为英文propmt.
    """
    # 检查是否有中文字符
    if any("\u4e00" <= char <= "\u9fff" for char in prompt):
        translated = translator(prompt)[0]["translation_text"]
        # 输出看看效果
        print(f"Translated prompt: {translated}")
        return translated
        # 添加一些通用的提示词
    return prompt


def generate_image(
    prompt,
    output_path="output.png",
    steps=20,
    guidance_scale=7.5,
    height=512,
    width=512,
):
    """
    参数:
        prompt (str): 图片的文本提示.
        output_path (str): 图片的输出路径.
        steps (int): DDPM的步骤 (通常在20-50之间).
        guidance_scale (float): CFG表示与prompt相似程度，一般2-5表示低相关性，10以上表示高相关性.

    Returns:
        str: 图片存储路径.
    """
    print(f"Generating image with prompt: {prompt}")
    print(f"Steps: {steps}, Guidance Scale: {guidance_scale}, Size: {height}x{width}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {pipe.device}")
    # 翻译prompt
    prompt = translate_prompt(prompt)

    # 生成图片
    start_time = time.time()
    image = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    ).images[0]

    # 保存图片
    os.makedirs("output", exist_ok=True)
    image.save(output_path)
    print(
        f"Image generated in {time.time() - start_time:.2f} seconds, saved to {output_path}"
    )
    return output_path


# -------------------------------------------入口--------------------------------------------
if __name__ == "__main__":

    prompts = [
        f"细胞有丝分裂的详细科学插图{tags['1']}{tags['4']}{tags['5']}",
        f"A vibrant scene of the Great Wall of China{tags['2']}{tags['3']}",
        f"A realistic depiction of a volcanic eruption{tags['1']}{tags['4']}{tags['5']}",
    ]

    steps = 50
    CFG = 8

    # 生成
    for i, prompt in enumerate(prompts):
        generate_image(
            prompt, f"output/{prompts[i]}.png", steps=steps, guidance_scale=CFG
        )

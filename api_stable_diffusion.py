from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import uuid
from stable_diffusion import generate_image, translate_prompt, pipe, translator, tags
from add_label import add_label

# -------------------------------------------初始化--------------------------------------------
app = FastAPI()

# 挂载静态文件目录，用于前端访问图像
app.mount("/images", StaticFiles(directory="output"), name="images")

# 启用CORS，允许React前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------数据模型--------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    selected_tags: list[str] = ["1", "4"]
    steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512


class LabelRequest(BaseModel):
    image_path: str
    label: str
    position_x: float
    position_y: float


# -------------------------------------------API端点--------------------------------------------
@app.get("/")
async def root():
    """
    根路径，返回欢迎信息
    """
    return {"message": "欢迎来到diffusion辅助教学图生成界面"}


@app.get("/tags")
async def get_tags():
    """
    返回可用的Tags，供前端复选框使用
    """
    return tags


@app.post("/generate")
async def api_generate_image(request: GenerateRequest):
    # 翻译Prompt
    prompt = translate_prompt(request.prompt)
    # 添加选中的Tags
    prompt += "".join(
        tags[tag_id] for tag_id in request.selected_tags if tag_id in tags
    )

    # 生成唯一文件名
    image_id = str(uuid.uuid4())[:8]
    output_path = f"output/image_{image_id}.png"
    image_url_path = f"/images/image_{image_id}.png"  # 返回给前端的路径

    try:
        generate_image(
            prompt,
            output_path,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            height=request.height,
            width=request.width,
        )
        return {"image_path": image_url_path}  # 返回 /images 路径
    except Exception as e:
        return {"error": f"生成图像失败: {str(e)}"}


@app.post("/label")
async def api_label_image(request: LabelRequest):
    """
    为指定图像添加标签

    Args:
        request: 包含image_path、label、position_x、position_y的请求体

    Returns:
        dict: 标注图像路径
    """
    # 基于输入image_path生成标注文件名
    image_id = (
        os.path.basename(request.image_path).replace("image_", "").replace(".png", "")
    )
    labeled_path = f"output/labeled_{image_id}.png"

    try:
        result_path = add_label(
            request.image_path,
            request.label,
            position=(request.position_x, request.position_y),
            output_path=labeled_path,
        )
        return {"labeled_path": result_path}
    except Exception as e:
        return {"error": f"添加标签失败: {str(e)}"}


# -------------------------------------------入口--------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from PIL import Image, ImageDraw, ImageFont
import os


def add_label(
    image_path, label, position=(50.0, 50.0), output_path="labeled_image.png"
):
    """
    给图片添加标签

    Args:
        image_path (str): 要处理的图片.
        label (str): 要添加标签的具体文本内容.
        position (tuple): 添加标签的坐标.
        output_path (str): 输出的位置.

    Returns:
        str: 输出的位置.
    """
    # 加载图片
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 设置字体信息
    try:
        font = ImageFont.truetype("arial.ttf", 24)  # Adjust size as needed
    except:
        font = ImageFont.load_default()

    # 添加标签
    draw.text(position, label, fill="black", font=font)
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Labeled image saved to {output_path}")
    return output_path


if __name__ == "__main__":
    add_label(
        "output/cell_mitosis.png", "Chromosome", (50, 50), "output/labeled_mitosis.png"
    )

import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs("assets", exist_ok=True)

img = Image.new("RGB", (512, 512), color=(34, 139, 34))  # 绿色背景
draw = ImageDraw.Draw(img)
draw.text((150, 240), "Sample Image", fill=(255, 255, 255))
img.save("assets/sample.jpg")

print("✅ Sample image saved to assets/sample.jpg")

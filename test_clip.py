import torch
import clip
from PIL import Image

# 加载CLIP模型（会自动下载，约1.8GB）
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 测试文本-图像匹配
image = preprocess(Image.open("test.jpg")).unsqueeze(0).to(device)  # 替换为任意图像路径
text = clip.tokenize(["a cat", "a dog"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("匹配概率：", probs)  # 输出类似[[0.99, 0.01]]即正常
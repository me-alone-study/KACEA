# KACEA

1. 创建新环境（推荐 Python 3.9，兼容性最佳）
# 新建名为"new_ibmea"的环境，指定Python 3.9
conda create -n new_ibmea python=3.9 -y
# 激活环境
conda activate new_ibmea
2. 配置渠道（确保下载速度和兼容性）
保留原环境的有效渠道，并补充必要渠道：

# 添加渠道（按优先级排序）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
conda config --add channels pytorch
conda config --set show_channel_urls yes  # 显示渠道来源
3. 安装核心依赖（PyTorch+CUDA）
CLIP 依赖 PyTorch 和 CUDA，需根据显卡型号选择适配的 CUDA 版本（以下以CUDA 11.7为例，支持大多数新显卡；旧显卡可换为 11.3）：

# 安装CUDA工具包（根据显卡选择：11.7支持RTX 30/40系列，11.3支持GTX 10/20系列）
conda install cudatoolkit=11.7 -y

# 安装PyTorch和配套库（对应CUDA 11.7的最新兼容版本）
 pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116   --extra-index-url https://download.pytorch.org/whl/cu116^C

验证 PyTorch 是否生效：
启动 Python，运行import torch; print(torch.cuda.is_available())，返回True则正常。
4. 安装 CLIP 及关键依赖
CLIP 需要直接从源码安装（或用 pip 安装），同时需补充其他依赖：

# 安装CLIP（OpenAI官方版本）
pip install git+https://github.com/openai/CLIP.git

# 安装CLIP依赖的辅助库（处理图像/文本）
pip install ftfy regex tqdm  # CLIP必需
pip install pillow==9.5.0  # 图像处理（适配高版本Python）
pip install numpy==1.24.3  # 数值计算（升级到3.9兼容版本）
5. 升级其他必要库（适配 Python 3.9）
同步更新常用库（确保与高版本 Python 兼容）：

# 数据处理库
conda install pandas==1.5.3 -y  # 升级pandas（支持Python 3.9）
conda install scikit-learn==1.2.2 -y  # 机器学习工具

# 深度学习辅助库
pip install transformers==4.30.2  # 升级transformers（支持CLIP相关模型）
pip install huggingface-hub==0.16.4  # 模型下载工具

# 可视化/调试库
conda install matplotlib==3.7.1 -y  # 绘图工具
pip install ipython==8.14.0  # 交互终端（适配3.9）
6. 验证环境是否可用
启动 Python，运行以下代码测试 CLIP：

运行
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

# 测试模型
python3 -u src/run.py \
    --file_dir data/DBP15K/zh_en \
    --epochs 3 \
    --check_point 1 \
    --pred_name test_complete \
    --bsize 500 \
    --hidden_units "32,32" \
    --structure_encoder gcn \
    --dropout 0.1 \
    --attr_dim 32 \
    --img_dim 32 \
    --char_dim 32 \
    --w_name \
    --w_char

# KACEA

1. 创建新环境（推荐 Python 3.9，兼容性最佳）
conda create -n KACEA python=3.9 -y

2. 配置渠道（确保下载速度和兼容性）
保留原环境的有效渠道，并补充必要渠道：
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
conda config --add channels pytorch
conda config --set show_channel_urls yes  # 显示渠道来源

4. 安装核心依赖（PyTorch+CUDA）

conda install cudatoolkit=11.6 -c nvidia -y --override-channels

# 安装PyTorch和配套库
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116

验证 PyTorch 是否生效：
启动 Python，运行

import torch; print(torch.cuda.is_available())

返回True则正常。
4. 安装 CLIP 及关键依赖
CLIP 需要直接从源码安装（或用 pip 安装），同时需补充其他依赖：

# 安装CLIP
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
conda install -c metric-learning pytorch-metric-learning
6. 验证环境是否可用
运行test_clip.py

  # 输出类似[[0.99, 0.01]]即正常

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

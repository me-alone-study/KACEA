# KACEA Installation Guide

This guide provides step-by-step instructions for setting up the KACEA environment with optimal compatibility using Python 3.9.

## 1. Create New Environment

Create a new conda environment with Python 3.9 for best compatibility:

```bash
conda create -n KACEA python=3.9 -y
conda activate KACEA
```

## 2. Configure Channels

Configure conda channels to ensure optimal download speed and compatibility:

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ 
conda config --add channels pytorch 
conda config --set show_channel_urls yes
```

## 3. Install Core Dependencies

### Install CUDA Toolkit

```bash
conda install cudatoolkit=11.6 -c nvidia -y --override-channels
```

### Install PyTorch with CUDA Support

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

### Verify PyTorch Installation

Start Python and run the following verification script:

```python
import torch
print(torch.cuda.is_available())
```

If the output is `True`, PyTorch with CUDA is properly installed.

## 4. Install CLIP and Key Dependencies

### Install CLIP

```bash
pip install git+https://github.com/openai/CLIP.git 
pip install ftfy regex tqdm 
```

### Install Image and Data Processing Libraries

```bash
pip install pillow==9.5.0  
pip install numpy==1.24.3 
conda install pandas==1.5.3 -y 
conda install scikit-learn==1.2.2 -y 
conda install matplotlib==3.7.1 -y  
pip install ipython==8.14.0 
```

### Install Machine Learning Libraries

```bash
pip install huggingface-hub==0.16.4 
conda install -c metric-learning pytorch-metric-learning
pip install pytorch-metric-learning tqdm scikit-learn
pip install transformers==4.35.2
pip install accelerate
```

## 5. Environment Verification

### Test CLIP Installation

Run the test script to verify CLIP is working properly:

```bash
python test_clip.py
```

Expected output should be similar to: `[[0.99, 0.01]]`

### Test Model Training

Run a complete model test with the following command:

```bash
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
```

## Dependencies Summary

| Package | Version | Installation Method |
|---------|---------|-------------------|
| Python | 3.9 | conda |
| PyTorch | 1.13.1+cu116 | pip |
| CUDA Toolkit | 11.6 | conda |
| NumPy | 1.24.3 | pip |
| Pandas | 1.5.3 | conda |
| Scikit-learn | 1.2.2 | conda |
| Transformers | 4.35.2 | pip |
| Pillow | 9.5.0 | pip |
| Matplotlib | 3.7.1 | conda |

## Troubleshooting

- If CUDA is not available, check your GPU drivers and CUDA installation
- For download issues, verify the conda channels are properly configured
- If package conflicts occur, try creating a fresh environment
- For Chinese mirror access issues, you may need to use default conda channels

## Notes

- This setup uses CUDA 11.6 for optimal compatibility
- Chinese Tsinghua mirrors are configured for faster downloads in China
- All version numbers are specified to ensure reproducible installations
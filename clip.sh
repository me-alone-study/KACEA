#!/bin/bash

# KACEA with CLIP - 运行脚本

echo "=== KACEA with CLIP Training Script ==="

# 检查GPU可用性
if nvidia-smi &> /dev/null; then
    echo "GPU available, using CUDA"
    CUDA_FLAG="--cuda"
else
    echo "No GPU found, using CPU"
    CUDA_FLAG=""
fi

# 基础参数
DATASET="data/DBP15K/zh_en"
EPOCHS=1000
BATCH_SIZE=7500
LR=0.005

# CLIP相关参数
CLIP_MODEL="openai/clip-vit-base-patch32"
CLIP_TEMPERATURE=0.07
CLIP_LAMBDA=0.1

echo "Dataset: $DATASET"
echo "CLIP Model: $CLIP_MODEL"
echo "Epochs: $EPOCHS"

# 方案1: 不使用CLIP（原版KACEA）
echo ""
echo "=== 方案1: 原版KACEA（不使用CLIP） ==="
python src/run.py \
    --file_dir $DATASET \
    --epochs $EPOCHS \
    --bsize $BATCH_SIZE \
    --lr $LR \
    --check_point 100 \
    $CUDA_FLAG

# 方案2: 使用CLIP但不使用实体名称
echo ""
echo "=== 方案2: KACEA + CLIP（仅图像特征） ==="
python src/run.py \
    --file_dir $DATASET \
    --epochs $EPOCHS \
    --bsize $BATCH_SIZE \
    --lr $LR \
    --check_point 100 \
    --use_clip \
    --clip_model_name $CLIP_MODEL \
    --clip_temperature $CLIP_TEMPERATURE \
    --clip_lambda $CLIP_LAMBDA \
    $CUDA_FLAG

# 方案3: 使用CLIP + 实体名称
echo ""
echo "=== 方案3: KACEA + CLIP + 实体名称 ==="
python src/run.py \
    --file_dir $DATASET \
    --epochs $EPOCHS \
    --bsize $BATCH_SIZE \
    --lr $LR \
    --check_point 100 \
    --use_clip \
    --use_entity_names \
    --clip_model_name $CLIP_MODEL \
    --clip_temperature $CLIP_TEMPERATURE \
    --clip_lambda $CLIP_LAMBDA \
    $CUDA_FLAG

# 方案4: 完整CLIP + 微调
echo ""
echo "=== 方案4: KACEA + CLIP + 微调 + VIB ==="
python src/run.py \
    --file_dir $DATASET \
    --epochs $EPOCHS \
    --bsize $BATCH_SIZE \
    --lr $LR \
    --check_point 100 \
    --use_clip \
    --use_entity_names \
    --clip_fine_tune \
    --use_clip_vib \
    --clip_model_name $CLIP_MODEL \
    --clip_temperature $CLIP_TEMPERATURE \
    --clip_lambda $CLIP_LAMBDA \
    --clip_fusion_strategy concat \
    $CUDA_FLAG

echo "=== 训练完成 ==="
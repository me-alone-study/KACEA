#!/bin/bash

# CLIP增强的IBMEA实验脚本
# run_clip_enhanced.sh

gpu_id='0'
dataset='zh_en'  # 'zh_en' 'ja_en' 'fr_en'
ratio=0.3
seed=2023
warm=200
joint_beta=3
ms_alpha=5
training_step=1500
bsize=7500

# CLIP相关参数
clip_model="openai/clip-vit-base-patch32"
clip_fusion="concat"  # concat, attention, gate
clip_lambda=0.1
clip_temp=0.07

if [[ "$dataset" == *"FB"* ]]; then
    dataset_dir='mmkb-datasets'
    tau=400
else
    dataset_dir='DBP15K'
    tau=0.1
    ratio=0.3
fi

echo "Running CLIP-enhanced IBMEA with dataset=${dataset}"
current_datetime=$(date +"%Y-%m-%d-%H-%M")
head_name=${current_datetime}_${dataset}_clip
file_name=${head_name}_${clip_fusion}_lambda${clip_lambda}

echo "Experiment name: ${file_name}"

CUDA_VISIBLE_DEVICES=${gpu_id} python3 -u src/run.py \
    --file_dir data/${dataset_dir}/${dataset} \
    --pred_name ${file_name} \
    --rate ${ratio} \
    --lr .006 \
    --epochs 1000 \
    --dropout 0.45 \
    --hidden_units "300,300,300" \
    --check_point 50  \
    --bsize ${bsize} \
    --il \
    --il_start 20 \
    --semi_learn_step 5 \
    --csls \
    --csls_k 3 \
    --seed ${seed} \
    --structure_encoder "gat" \
    --img_dim 100 \
    --attr_dim 100 \
    --use_nce \
    --tau ${tau} \
    --use_sheduler_cos \
    --num_warmup_steps ${warm} \
    --num_training_steps ${training_step} \
    --joint_beta ${joint_beta} \
    --ms_alpha ${ms_alpha} \
    --w_name \
    --w_char \
    --use_clip \
    --clip_model_name ${clip_model} \
    --clip_fusion_strategy ${clip_fusion} \
    --clip_lambda ${clip_lambda} \
    --clip_temperature ${clip_temp} \
    --use_entity_names \
    > logs/${file_name}.log 2>&1

echo "Experiment completed. Check logs/${file_name}.log for results."
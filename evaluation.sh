#!/bin/bash

# ====================================================================
# IBMEA 统一快速验证脚本
# 基于你的DBP15K和MMKB两个成功脚本
# ====================================================================

GPU_ID='0'
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOGS_DIR="logs/unified_eval_${TIMESTAMP}"
mkdir -p ${LOGS_DIR}

echo "开始IBMEA统一快速验证 (基于你的原始成功配置)..."

# ====================================================================
# 1. DBP15K数据集测试 - 使用你的DBP15K参数
# ====================================================================

echo "1. 运行DBP15K数据集测试..."

# DBP15K参数 - 完全来自你的脚本
dbp_datasets=("zh_en" "ja_en" "fr_en")

for dataset in "${dbp_datasets[@]}"; do
    # 设置DBP15K参数
    ratio=0.3
    seed=2023
    warm=200
    joint_beta=3
    ms_alpha=5
    training_step=1500
    bsize=7500
    dataset_dir='DBP15K'
    tau=0.1
    
    echo "处理DBP15K数据集: ${dataset}"
    
    # 1.1 基准版本 (无CLIP)
    current_datetime=$(date +"%Y-%m-%d-%H-%M")
    head_name=${current_datetime}_${dataset}_baseline
    file_name=${head_name}_bsize${bsize}_${ratio}
    
    echo "运行基准: ${file_name}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u src/run.py \
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
        > ${LOGS_DIR}/${file_name}.log 2>&1
    
    echo "完成基准: ${file_name}"
    
    # 1.2 CLIP增强版本
    current_datetime=$(date +"%Y-%m-%d-%H-%M")
    head_name=${current_datetime}_${dataset}_clip
    file_name=${head_name}_bsize${bsize}_${ratio}
    
    echo "运行CLIP增强: ${file_name}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u src/run.py \
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
        --clip_model_name openai/clip-vit-base-patch32 \
        --clip_lambda 0.1 \
        --clip_temperature 0.07 \
        --clip_fusion_strategy concat \
        --use_entity_names \
        > ${LOGS_DIR}/${file_name}.log 2>&1
    
    echo "完成CLIP增强: ${file_name}"
done

# ====================================================================
# 2. MMKB数据集测试 - 使用你的MMKB参数
# ====================================================================

echo "2. 运行MMKB数据集测试..."

# MMKB参数 - 完全来自你的脚本
mmkb_datasets=("FB15K_DB15K" "FB15K_YAGO15K")

for dataset in "${mmkb_datasets[@]}"; do
    # 设置MMKB参数
    ratio=0.2
    seed=2023
    warm=400
    bsize=1000
    dataset_dir='mmkb-datasets'
    tau=400
    
    echo "处理MMKB数据集: ${dataset}"
    
    # 2.1 基准版本 (无CLIP)
    current_datetime=$(date +"%Y-%m-%d-%H-%M")
    head_name=${current_datetime}_${dataset}_baseline
    file_name=${head_name}_bsize${bsize}_${ratio}
    
    echo "运行基准: ${file_name}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u src/run.py \
        --file_dir data/${dataset_dir}/${dataset} \
        --pred_name ${file_name} \
        --rate ${ratio} \
        --lr .006 \
        --epochs 500 \
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
        --w_name \
        --w_char \
        > ${LOGS_DIR}/${file_name}.log 2>&1
    
    echo "完成基准: ${file_name}"
    
    # 2.2 CLIP增强版本
    current_datetime=$(date +"%Y-%m-%d-%H-%M")
    head_name=${current_datetime}_${dataset}_clip
    file_name=${head_name}_bsize${bsize}_${ratio}
    
    echo "运行CLIP增强: ${file_name}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u src/run.py \
        --file_dir data/${dataset_dir}/${dataset} \
        --pred_name ${file_name} \
        --rate ${ratio} \
        --lr .006 \
        --epochs 500 \
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
        --w_name \
        --w_char \
        --use_clip \
        --clip_model_name openai/clip-vit-base-patch32 \
        --clip_lambda 0.1 \
        --clip_temperature 0.07 \
        --clip_fusion_strategy concat \
        --use_entity_names \
        > ${LOGS_DIR}/${file_name}.log 2>&1
    
    echo "完成CLIP增强: ${file_name}"
done

# ====================================================================
# 3. 快速消融研究 (仅在zh_en上，使用较短epoch)
# ====================================================================

echo "3. 运行快速消融研究 (zh_en)..."

dataset="zh_en"
ratio=0.3
seed=2023
warm=100  # 减少warmup
joint_beta=3
ms_alpha=5
training_step=300  # 减少训练步数
bsize=5000  # 稍小batch size
dataset_dir='DBP15K'
tau=0.1

# 消融研究基础配置 (快速版本)
ABLATION_BASE="--file_dir data/${dataset_dir}/${dataset} --rate ${ratio} --lr .006 --epochs 300 --dropout 0.45 --hidden_units 300,300,300 --check_point 25 --bsize ${bsize} --il --il_start 20 --semi_learn_step 5 --csls --csls_k 3 --seed ${seed} --structure_encoder gat --img_dim 100 --attr_dim 100 --use_nce --tau ${tau} --use_sheduler_cos --num_warmup_steps ${warm} --num_training_steps ${training_step} --joint_beta ${joint_beta} --ms_alpha ${ms_alpha}"

# 3.1 仅图像模态
current_datetime=$(date +"%Y-%m-%d-%H-%M")
file_name=${current_datetime}_ablation_img_only

echo "运行消融 - 仅图像: ${file_name}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u src/run.py \
    ${ABLATION_BASE} \
    --pred_name ${file_name} \
    --w_rel --w_attr --w_name --w_char \
    --use_clip \
    --clip_model_name openai/clip-vit-base-patch32 \
    --clip_lambda 0.1 \
    --clip_temperature 0.07 \
    --clip_fusion_strategy concat \
    --use_entity_names \
    > ${LOGS_DIR}/${file_name}.log 2>&1

# 3.2 仅文本/属性模态
current_datetime=$(date +"%Y-%m-%d-%H-%M")
file_name=${current_datetime}_ablation_text_only

echo "运行消融 - 仅文本: ${file_name}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u src/run.py \
    ${ABLATION_BASE} \
    --pred_name ${file_name} \
    --w_img --w_rel --w_name --w_char \
    --use_clip \
    --clip_model_name openai/clip-vit-base-patch32 \
    --clip_lambda 0.1 \
    --clip_temperature 0.07 \
    --clip_fusion_strategy concat \
    --use_entity_names \
    > ${LOGS_DIR}/${file_name}.log 2>&1

# 3.3 无CLIP完整模态
current_datetime=$(date +"%Y-%m-%d-%H-%M")
file_name=${current_datetime}_ablation_no_clip

echo "运行消融 - 无CLIP: ${file_name}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u src/run.py \
    ${ABLATION_BASE} \
    --pred_name ${file_name} \
    --w_name --w_char \
    > ${LOGS_DIR}/${file_name}.log 2>&1

# ====================================================================
# 4. CLIP融合策略对比 (仅在zh_en上)
# ====================================================================

echo "4. 运行CLIP融合策略对比..."

strategies=("concat" "attention" "gate")

for strategy in "${strategies[@]}"; do
    current_datetime=$(date +"%Y-%m-%d-%H-%M")
    file_name=${current_datetime}_fusion_${strategy}
    
    echo "运行融合策略 - ${strategy}: ${file_name}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u src/run.py \
        ${ABLATION_BASE} \
        --pred_name ${file_name} \
        --w_name --w_char \
        --use_clip \
        --clip_model_name openai/clip-vit-base-patch32 \
        --clip_lambda 0.1 \
        --clip_temperature 0.07 \
        --clip_fusion_strategy ${strategy} \
        --use_entity_names \
        > ${LOGS_DIR}/${file_name}.log 2>&1
    
    echo "完成融合策略 - ${strategy}: ${file_name}"
done

# ====================================================================
# 5. 结果统计和报告生成
# ====================================================================

echo "5. 生成结果统计和报告..."

cat > ${LOGS_DIR}/extract_unified_results.py << 'EOF'
#!/usr/bin/env python3
import os
import re
import pandas as pd
import glob
from datetime import datetime

def extract_result(log_file):
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        pattern = r'best epoch is (\d+), hits@1 hits@10 MRR MR is: \[([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\]'
        match = re.search(pattern, content)
        
        if match:
            filename = os.path.basename(log_file).replace('.log', '')
            # 解析文件名以提取信息
            parts = filename.split('_')
            
            return {
                'experiment': filename,
                'dataset': parts[3] if len(parts) > 3 else 'unknown',
                'type': parts[4] if len(parts) > 4 else 'unknown',
                'best_epoch': int(match.group(1)),
                'hits_1': float(match.group(2)),
                'hits_10': float(match.group(3)),
                'mrr': float(match.group(4)),
                'mr': float(match.group(5))
            }
    except Exception as e:
        print(f"Error: {log_file} - {e}")
    return None

def main():
    results = []
    log_dir = os.path.dirname(os.path.abspath(__file__))
    
    for log_file in glob.glob(f"{log_dir}/*.log"):
        result = extract_result(log_file)
        if result:
            results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('hits_1', ascending=False)
        
        # 保存完整结果
        df.to_csv(f"{log_dir}/unified_results.csv", index=False)
        
        # 生成详细报告
        with open(f"{log_dir}/unified_report.txt", 'w') as f:
            f.write("IBMEA 统一验证结果报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总实验数: {len(results)}\n\n")
            
            # DBP15K结果分析
            dbp_results = df[df['dataset'].isin(['zh', 'ja', 'fr'])]
            if not dbp_results.empty:
                f.write("📊 DBP15K数据集结果:\n")
                
                # 基准 vs CLIP对比
                for lang in ['zh', 'ja', 'fr']:
                    lang_data = dbp_results[dbp_results['dataset'] == lang]
                    baseline = lang_data[lang_data['type'] == 'baseline']
                    clip = lang_data[lang_data['type'] == 'clip']
                    
                    if not baseline.empty and not clip.empty:
                        baseline_score = baseline['hits_1'].iloc[0]
                        clip_score = clip['hits_1'].iloc[0]
                        improvement = clip_score - baseline_score
                        f.write(f"  {lang}_en:\n")
                        f.write(f"    基准:     {baseline_score:.4f}\n")
                        f.write(f"    CLIP:     {clip_score:.4f}\n")
                        f.write(f"    提升:     {improvement:.4f} ({improvement/baseline_score*100:.1f}%)\n")
                
                # DBP15K平均提升
                dbp_baseline = dbp_results[dbp_results['type'] == 'baseline']
                dbp_clip = dbp_results[dbp_results['type'] == 'clip']
                if not dbp_baseline.empty and not dbp_clip.empty:
                    avg_baseline = dbp_baseline['hits_1'].mean()
                    avg_clip = dbp_clip['hits_1'].mean()
                    avg_improvement = avg_clip - avg_baseline
                    f.write(f"\n  📈 DBP15K平均提升: {avg_improvement:.4f} ({avg_improvement/avg_baseline*100:.1f}%)\n\n")
            
            # MMKB结果分析
            mmkb_results = df[df['dataset'].isin(['FB15K', 'YAGO15K']) | df['experiment'].str.contains('FB15K')]
            if not mmkb_results.empty:
                f.write("📊 MMKB数据集结果:\n")
                
                mmkb_baseline = mmkb_results[mmkb_results['type'] == 'baseline']
                mmkb_clip = mmkb_results[mmkb_results['type'] == 'clip']
                
                if not mmkb_baseline.empty and not mmkb_clip.empty:
                    avg_baseline = mmkb_baseline['hits_1'].mean()
                    avg_clip = mmkb_clip['hits_1'].mean()
                    avg_improvement = avg_clip - avg_baseline
                    f.write(f"  基准平均: {avg_baseline:.4f}\n")
                    f.write(f"  CLIP平均: {avg_clip:.4f}\n")
                    f.write(f"  📈 平均提升: {avg_improvement:.4f} ({avg_improvement/avg_baseline*100:.1f}%)\n\n")
            
            # 消融研究结果
            ablation_results = df[df['experiment'].str.contains('ablation')]
            if not ablation_results.empty:
                f.write("🧪 消融研究结果:\n")
                for _, row in ablation_results.iterrows():
                    exp_type = row['experiment'].split('_')[-2:]
                    exp_name = '_'.join(exp_type)
                    f.write(f"  {exp_name:15}: {row['hits_1']:.4f}\n")
                f.write("\n")
            
            # 融合策略对比
            fusion_results = df[df['experiment'].str.contains('fusion')]
            if not fusion_results.empty:
                f.write("🔀 CLIP融合策略对比:\n")
                best_fusion = fusion_results.loc[fusion_results['hits_1'].idxmax()]
                for _, row in fusion_results.iterrows():
                    strategy = row['experiment'].split('_')[-1]
                    marker = "⭐" if row['experiment'] == best_fusion['experiment'] else "  "
                    f.write(f"  {marker} {strategy:10}: {row['hits_1']:.4f}\n")
                f.write(f"\n  🏆 最佳策略: {best_fusion['experiment'].split('_')[-1]}\n\n")
            
            # Top 10结果
            f.write("🏆 Top 10 实验结果:\n")
            for i, (_, row) in enumerate(df.head(10).iterrows()):
                dataset_name = row['dataset'] if len(row['dataset']) < 15 else row['dataset'][:12] + "..."
                f.write(f"  {i+1:2}. {dataset_name:15} {row['type']:10}: {row['hits_1']:.4f}\n")
            
            # 关键发现
            f.write("\n💡 关键发现:\n")
            if not dbp_results.empty:
                best_dbp = dbp_results.loc[dbp_results['hits_1'].idxmax()]
                f.write(f"  - DBP15K最佳: {best_dbp['dataset']}_en {best_dbp['type']} = {best_dbp['hits_1']:.4f}\n")
            
            if not mmkb_results.empty:
                best_mmkb = mmkb_results.loc[mmkb_results['hits_1'].idxmax()]
                f.write(f"  - MMKB最佳: {best_mmkb['hits_1']:.4f}\n")
            
            overall_best = df.loc[df['hits_1'].idxmax()]
            f.write(f"  - 整体最佳: {overall_best['experiment']} = {overall_best['hits_1']:.4f}\n")
        
        print(f"\n✅ 统一验证完成！处理了 {len(results)} 个实验")
        print("📁 结果文件:")
        print(f"   - {log_dir}/unified_results.csv")
        print(f"   - {log_dir}/unified_report.txt")
        
        # 核心结果概览
        print("\n🏆 === 核心结果概览 ===")
        
        # DBP15K概览
        dbp_results = df[df['dataset'].isin(['zh', 'ja', 'fr'])]
        if not dbp_results.empty:
            dbp_baseline = dbp_results[dbp_results['type'] == 'baseline']['hits_1'].mean()
            dbp_clip = dbp_results[dbp_results['type'] == 'clip']['hits_1'].mean()
            print(f"📊 DBP15K: 基准={dbp_baseline:.4f}, CLIP={dbp_clip:.4f}, 提升={dbp_clip-dbp_baseline:.4f}")
        
        # MMKB概览
        mmkb_results = df[df['experiment'].str.contains('FB15K')]
        if not mmkb_results.empty:
            mmkb_baseline = mmkb_results[mmkb_results['type'] == 'baseline']['hits_1'].mean()
            mmkb_clip = mmkb_results[mmkb_results['type'] == 'clip']['hits_1'].mean()
            print(f"📊 MMKB:   基准={mmkb_baseline:.4f}, CLIP={mmkb_clip:.4f}, 提升={mmkb_clip-mmkb_baseline:.4f}")
        
        # 最佳结果
        best = df.loc[df['hits_1'].idxmax()]
        print(f"🏆 最佳:   {best['experiment'][:30]}... = {best['hits_1']:.4f}")
            
    else:
        print("❌ 未找到有效结果")

if __name__ == '__main__':
    main()
EOF

echo "等待5秒后开始结果统计..."
sleep 5
python3 ${LOGS_DIR}/extract_unified_results.py

echo "======================================================================"
echo "🎉 统一验证完成！"
echo "📍 日志位置: ${LOGS_DIR}"
echo "📊 查看详细报告: cat ${LOGS_DIR}/unified_report.txt"
echo "📈 查看CSV结果: ${LOGS_DIR}/unified_results.csv"
echo "======================================================================"
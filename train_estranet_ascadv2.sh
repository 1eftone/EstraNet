#!/bin/bash

# --- 1. Slurm 资源配置 ---
#SBATCH --job-name=Estranet_train
#SBATCH --output=training_log_%j.out
#SBATCH --error=training_error_%j.err
#SBATCH --gpus=6000ada:1
#SBATCH --time=24:00:00  
# --- 2. 打印作业信息 ---
echo "=========================================================="
echo "Job Started at $(date)"
echo "Job running on node $(hostname)"
echo "=========================================================="

# --- 3. 设置 "黄金工作流程" 环境 ---
# 激活 Conda 环境
module load Miniforge3
#source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
source activate estranet_final

# 加载官方模块
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

# # 设置 XLA 环境变量
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

# --- 4. 运行您的程序脚本 ---
# 这一行会调用您帖子中提供的那个脚本，并传入 "train" 参数
bash run_trans_ascadv2.sh train

# --- 5. 打印结束信息 ---
echo "=========================================================="
echo "Job Finished at $(date)"
echo "=========================================================="
#!/bin/bash

# --- 自动使用当前激活环境的 Python ---
# 只要你在 sbatch 里激活了 estranet_final，这里就会自动用对的
PYTHON_EXEC="python"

# 打印一下验证，你会看到它指向 estranet_final
echo "Which python: $(which python)"
echo "Using python: $PYTHON_EXEC"
# GPU config
USE_TPU=False

# Experiment config
# 建议使用绝对路径，避免找不到文件
DATA_PATH=/home/e240023/EstraNet/datasets/ascadv2-extracted.h5
DATASET=ASCADv2
CKP_DIR=./checkpoints_ascadv2
CKP_IDX=0
WARM_START=False
RESULT_PATH=results_ascadv2

# Optimization config
LEARNING_RATE=2.5e-4
CLIP=0.25
MIN_LR_RATIO=0.004

# ASCADv2 Config
INPUT_LENGTH=15000 
DATA_DESYNC=200 

# Training config
TRAIN_BSZ=16
EVAL_BSZ=16
TRAIN_STEPS=400000
WARMUP_STEPS=10000
ITERATIONS=5000
SAVE_STEPS=10000

# Model config
N_LAYER=2
D_MODEL=128
D_HEAD=32
N_HEAD=8
D_INNER=256
N_HEAD_SM=8
D_HEAD_SM=16
DROPOUT=0.05
CONV_KERNEL_SIZE=3 
N_CONV_LAYER=2
POOL_SIZE=10
D_KERNEL_MAP=512
BETA_HAT_2=225

MODEL_NORM='preLC'
HEAD_INIT='forward'
SM_ATTN=True

# Evaluation config
MAX_EVAL_BATCH=100
OUTPUT_ATTN=False

if [[ $1 == 'train' ]]; then
    python train_trans.py \
        --use_tpu=${USE_TPU} \
        --data_path=${DATA_PATH} \
        --dataset=${DATASET} \
        --checkpoint_dir=${CKP_DIR} \
        --warm_start=${WARM_START} \
        --result_path=${RESULT_PATH} \
        --learning_rate=${LEARNING_RATE} \
        --clip=${CLIP} \
        --min_lr_ratio=${MIN_LR_RATIO} \
        --warmup_steps=${WARMUP_STEPS} \
        --input_length=${INPUT_LENGTH} \
        --data_desync=${DATA_DESYNC} \
        --train_batch_size=${TRAIN_BSZ} \
        --eval_batch_size=${EVAL_BSZ} \
        --train_steps=${TRAIN_STEPS} \
        --iterations=${ITERATIONS} \
        --save_steps=${SAVE_STEPS} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_head=${D_HEAD} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --n_head_softmax=${N_HEAD_SM} \
        --d_head_softmax=${D_HEAD_SM} \
        --dropout=${DROPOUT} \
        --conv_kernel_size=${CONV_KERNEL_SIZE} \
        --n_conv_layer=${N_CONV_LAYER} \
        --pool_size=${POOL_SIZE} \
        --d_kernel_map=${D_KERNEL_MAP} \
        --beta_hat_2=${BETA_HAT_2} \
        --model_normalization=${MODEL_NORM} \
        --head_initialization=${HEAD_INIT} \
        --softmax_attn=${SM_ATTN} \
        --max_eval_batch=${MAX_EVAL_BATCH} \
        --do_train=True
elif [[ $1 == 'test' ]]; then
    python train_trans.py \
        --use_tpu=${USE_TPU} \
        --data_path=${DATA_PATH} \
        --dataset=${DATASET} \
        --checkpoint_dir=${CKP_DIR} \
        --checkpoint_idx=${CKP_IDX} \
        --warm_start=${WARM_START} \
        --result_path=${RESULT_PATH} \
        --learning_rate=${LEARNING_RATE} \
        --clip=${CLIP} \
        --min_lr_ratio=${MIN_LR_RATIO} \
        --warmup_steps=${WARMUP_STEPS} \
        --input_length=${INPUT_LENGTH} \
        --train_batch_size=${TRAIN_BSZ} \
        --eval_batch_size=${EVAL_BSZ} \
        --train_steps=${TRAIN_STEPS} \
        --iterations=${ITERATIONS} \
        --save_steps=${SAVE_STEPS} \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_head=${D_HEAD} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --n_head_softmax=${N_HEAD_SM} \
        --d_head_softmax=${D_HEAD_SM} \
        --dropout=${DROPOUT} \
        --conv_kernel_size=${CONV_KERNEL_SIZE} \
        --n_conv_layer=${N_CONV_LAYER} \
        --pool_size=${POOL_SIZE} \
        --d_kernel_map=${D_KERNEL_MAP} \
        --beta_hat_2=${BETA_HAT_2} \
        --model_normalization=${MODEL_NORM} \
        --head_initialization=${HEAD_INIT} \
        --softmax_attn=${SM_ATTN} \
        --max_eval_batch=${MAX_EVAL_BATCH} \
        --output_attn=${OUTPUT_ATTN} \
        --do_train=False
else
    echo "unknown argument 1"
fi
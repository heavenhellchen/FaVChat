#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../r1-v"

export DEBUG_MODE="true"
export LOG_PATH="./debug_favchat_sft.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4"     --nnodes="1"     --node_rank="0"     --master_addr="127.0.0.1"     --master_port="12349"     --module open_r1.sft_favchat     --output_dir "./log/Qwen2.5-VL-7B-FaVChat-SFT"     --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct"     --dataset_name "./FaVChat-data/FaVChat-COT-170k.json"     --data_root "./FaVChat-data"     --deepspeed local_scripts/zero2.json     --per_device_train_batch_size 1     --gradient_accumulation_steps 2     --learning_rate 1e-6     --logging_steps 1     --bf16     --report_to wandb     --gradient_checkpointing true     --attn_implementation flash_attention_2     --num_train_epochs 1     --run_name Qwen2.5-VL-7B-FaVChat-SFT     --save_steps 1000     --max_grad_norm 5     --save_only_model true

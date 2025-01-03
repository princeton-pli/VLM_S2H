#!/bin/bash
#SBATCH --gres=
#SBATCH --mem=
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-gpu
#SBATCH --output=logs/%x-%j-%a.out
#SBATCH --partition
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --job-name=pretrain-eagle-x2-llama3-8b
#SBATCH -t 

conda activate VLM_S2H

export WANDB_MODE=offline
export WANDB_DISABLED="true"

GLOBAL_BATCH_SIZE=256 ## SHOULD BE SAME FOR ALL PRETRAINING
BATCH_SIZE_PER_DEVICE=8 ## WORKS FOR LLAMA3
N_GPUS="$(( $(echo $SLURM_JOB_GPUS| grep -o , | wc -l) + 1 ))"
ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / BATCH_SIZE_PER_DEVICE / N_GPUS))

NAME=pretrain-eagle-x2-llama3-8b

cd EAGLE
deepspeed --master_port 32004 ./train_mem.py \
    --deepspeed ../scripts/zero2.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --version llava_llama_3 \
    --data_path ../pretraining_data/chat.json \
    --image_folder ../pretraining_data/LLaVA-CC3M-Pretrain-595K \
    --vision_tower "clip-448;convnext-1024" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ../checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_SIZE_PER_DEVICE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 24000 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True
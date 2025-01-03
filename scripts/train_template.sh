#!/bin/bash
#SBATCH --gres=gpu:h100:8
#SBATCH --partition=
#SBATCH --mem=1000GB
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-gpu 12
#SBATCH --output=logs/%x-%j-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --job-name=
#SBATCH -t 01:00:01
#SBATCH --array=

conda activate VLM_S2H

export WANDB_MODE=offline
export WANDB_DISABLED="true"

GLOBAL_BATCH_SIZE=128 ## SHOULD BE SAME FOR ALL FINETUNING
BATCH_SIZE_PER_DEVICE=1 ## WORKS FOR LLAMA3
N_GPUS="$(( $(echo $SLURM_JOB_GPUS| grep -o , | wc -l) + 1 ))"
ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / BATCH_SIZE_PER_DEVICE / N_GPUS))

OPTION=$SLURM_ARRAY_TASK_ID

MODEL=llama3-8b
VERSION=llava_llama_3
echo "MODEL: $MODEL"
echo "VERSION: $VERSION"

NUM_DATA=${num_data:-60}
N_EPOCHS=1

if [[ $OPTION == 1 ]]; then
    N_EPOCHS=3
fi
if [[ $OPTION == 2 ]]; then
    N_EPOCHS=2
fi
if [[ $OPTION == 3 ]]; then
    N_EPOCHS=2
fi
if [[ $OPTION == 4 ]]; then
    N_EPOCHS=3
fi
if [[ $OPTION == 5 ]]; then
    N_EPOCHS=1.5
fi

### IF EXPERIMENTING WITH NO_COT, SHOULD CALL 
### export no_cot=_noCoT
### FIRST
COT_TAG=${no_cot:-}
ADD_TAG=${add_tag:-}
INTERNALIZE_TAG=${internalize_cot_ar:-}
if [[ $sequential == "True" ]]; then
    SEQUENTIAL_TAG="_sequential"
fi

echo "GETTING THE DATA PATH"
DATA_PATH="$(python ./scripts/combine_data.py --setting ${SETTING} --option ${OPTION} --total_num_data ${NUM_DATA})"
if [[ $COT_TAG == "_noCoT" ]]; then
    DATA_PATH="$(python ./scripts/combine_data.py --setting ${SETTING} --option ${OPTION} --total_num_data ${NUM_DATA} --no_cot)"
fi
if [[ $sequential == "True" ]]; then
    DATA_PATH="$(python ./scripts/combine_data.py --setting ${SETTING} --option ${OPTION} --total_num_data ${NUM_DATA} --sequential)"
fi
if [[ $INTERNALIZE_TAG == "_interncot" ]]; then
    DATA_PATH="$(python ./scripts/combine_data.py --setting ${SETTING} --option ${OPTION} --total_num_data ${NUM_DATA} --internalize_cot_ar)"
fi
DATA_PATH="$(echo "$DATA_PATH" | tail -1)"
echo "DATA_PATH: $DATA_PATH"

NAME=continual-finetune-eagle-x2-${MODEL}-1.8M-${SETTING_CAPITAL}-Option${OPTION}-${NUM_DATA}k${COT_TAG}${INTERNALIZE_TAG}${SEQUENTIAL_TAG}${ADD_TAG}
echo "MODEL NAME: $NAME"

cd EAGLE

### 234 IS ROUGHLY 30K EXAMPLES
deepspeed --master_port 31008 ./train_mem.py \
    --deepspeed ../scripts/zero2.json \
    --model_name_or_path ../checkpoints/finetune-eagle-x2-${MODEL}-1.8M \
    --version ${VERSION} \
    --data_path ../${DATA_PATH} \
    --image_folder ../pretraining_data/Eagle-1.8M \
    --vision_tower "clip-448;convnext-1024" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ../checkpoints/$NAME \
    --num_train_epochs ${N_EPOCHS} \
    --per_device_train_batch_size $BATCH_SIZE_PER_DEVICE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 234 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True
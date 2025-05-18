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
#SBATCH -t 02:00:00
#SBATCH --array=

conda activate VLM_S2H_qwen

export WANDB_MODE=offline

GLOBAL_BATCH_SIZE=128 ## SHOULD BE SAME FOR ALL FINETUNING
BATCH_SIZE_PER_DEVICE=4 ## WORKS FOR 7B
N_GPUS_PER_NODE="$(( $(echo $SLURM_JOB_GPUS| grep -o , | wc -l) + 1 ))"
TOTAL_N_GPUS=$((N_GPUS_PER_NODE * SLURM_NNODES))
ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / BATCH_SIZE_PER_DEVICE / TOTAL_N_GPUS))

OPTION=$SLURM_ARRAY_TASK_ID

MODEL=Qwen2.5-VL-7B-Instruct
echo "MODEL: $MODEL"

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
if [[ $OPTION == 10 ]]; then
    DATA_PATH="$(python ./scripts_qwen/combine_data_medium.py --setting ${SETTING} --option ${OPTION} --total_num_data ${NUM_DATA})"
else
    DATA_PATH="$(python ./scripts_qwen/combine_data.py --setting ${SETTING} --option ${OPTION} --total_num_data ${NUM_DATA})"
    if [[ $COT_TAG == "_noCoT" ]]; then
        DATA_PATH="$(python ./scripts_qwen/combine_data.py --setting ${SETTING} --option ${OPTION} --total_num_data ${NUM_DATA} --no_cot)"
    fi
    if [[ $sequential == "True" ]]; then
        DATA_PATH="$(python ./scripts_qwen/combine_data.py --setting ${SETTING} --option ${OPTION} --total_num_data ${NUM_DATA} --sequential)"
    fi
    if [[ $INTERNALIZE_TAG == "_interncot" ]]; then
        DATA_PATH="$(python ./scripts_qwen/combine_data.py --setting ${SETTING} --option ${OPTION} --total_num_data ${NUM_DATA} --internalize_cot_ar)"
    fi
fi
DATA_PATH="$(echo "$DATA_PATH" | tail -1)"
echo "DATA_PATH: $DATA_PATH"

NAME=continual-finetune-${MODEL}-${SETTING_CAPITAL}-Option${OPTION}-${NUM_DATA}k${COT_TAG}${INTERNALIZE_TAG}${SEQUENTIAL_TAG}${ADD_TAG}
echo "MODEL NAME: $NAME"

cd scripts_qwen

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo $MASTER_ADDR
echo $MASTER_PORT
echo $SLURM_NNODES
echo $TOTAL_N_GPUS

### 234 IS ROUGHLY 30K EXAMPLES
accelerate launch \
    --config_file ./fsdp_config.yaml \
    --num_machines $SLURM_NNODES \
    --num_processes $TOTAL_N_GPUS \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    train.py \
        --model_name_or_path Qwen/${MODEL} \
        --data_path ../${DATA_PATH} \
        --output_dir ../checkpoints/$NAME \
        --num_train_epochs ${N_EPOCHS} \
        --per_device_train_batch_size $BATCH_SIZE_PER_DEVICE \
        --gradient_accumulation_steps $ACCUM_STEPS \
        --save_steps 234 \
        --learning_rate 2e-5

cp Qwen/${MODEL}/chat_template.json ../checkpoints/$NAME/chat_template.json
cp Qwen/${MODEL}/preprocessor_config.json ../checkpoints/$NAME/preprocessor_config.json

chmod -R 770 ../checkpoints/$NAME

TRAINING_COMPLETE=true
for i in {1..7}
do
    if [[ ! -f "../checkpoints/$NAME/model-0000${i}-of-00007.safetensors" ]]; then
    TRAINING_COMPLETE=false
    fi
done

if [ "$TRAINING_COMPLETE" = true ]; then
    rm -rf ../checkpoints/$NAME/checkpoint-*
else
    exit 1
fi
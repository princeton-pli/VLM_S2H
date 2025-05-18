#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=
#SBATCH --mem=125GB
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-gpu 12
#SBATCH --output=logs/%x-%j-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH -t 28:00:00
#SBATCH --job-name=
#SBATCH --array=
#SBATCH --dependency=""

conda activate VLM_S2H_qwen

OPTION=$SLURM_ARRAY_TASK_ID

MODEL=Qwen2.5-VL-7B-Instruct
echo "MODEL: $MODEL"

NUM_DATA=${num_data:-60}

COT_TAG=${no_cot:-}
ADD_TAG=${add_tag:-}
INTERNALIZE_TAG=${internalize_cot_ar:-}

NAME=continual-finetune-${MODEL}-${SETTING_CAPITAL}-Option${OPTION}-${NUM_DATA}k${COT_TAG}${INTERNALIZE_TAG}${SEQUENTIAL_TAG}${ADD_TAG}

cd VLMEvalKit

python run.py --model $NAME --data OurEval_${SETTING_CAPITAL}_SIMPLE_text
python run.py --model $NAME --data OurEval_${SETTING_CAPITAL}_SIMPLE_image

if [[ ${SETTING_CAPITAL} == "ConsecutiveTableReadout" ]]; then
python run.py --model $NAME --data OurEval_${SETTING_CAPITAL}_MEDIUM_text
python run.py --model $NAME --data OurEval_${SETTING_CAPITAL}_MEDIUM_image
fi

python run.py --model $NAME --data OurEval_${SETTING_CAPITAL}_HARD_text
python run.py --model $NAME --data OurEval_${SETTING_CAPITAL}_HARD_image
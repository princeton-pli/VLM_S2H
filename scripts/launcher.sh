# EXAMPLE RUN:
# source scripts/launcher.sh 0 1,2,3,4 60
# THIS WILL RUN CONSECUTIVE_TABLE_READOUT WITH OPTION 1,2,3,4 ON 60K EXAMPLES

# (OPTIONAL) RUN THESE COMMANDS IN THE BASH BEFORE RUNNING THIS LAUNCHER
# export no_cot=_noCoT
# export sequential=True

SETTING_INDEX=${1}      ## 0 for consecutive_table_readout, 1 for table_readout, 2 for grid_navigation, 3 for visual_analogy
OPTION_INDICES=${2}     ## Comma separated with no space (e.g., 1,2,4)
export num_data=${3}    ## 60, 120, 240, ...
export add_tag=${4}
export no_cot=${5}

ALL_SETTINGS=(consecutive_table_readout table_readout grid_navigation visual_analogy)
ALL_SETTINGS_CAPITAL=(ConsecutiveTableReadout TableReadout GridNavigation VisualAnalogy)

export SETTING=${ALL_SETTINGS[$SETTING_INDEX]}
export SETTING_CAPITAL=${ALL_SETTINGS_CAPITAL[$SETTING_INDEX]}

## SUGGESTED TIMES FOR 8 H100 GPUS
if [[ $num_data == 30 ]]; then
    TRAIN_TIME="01:20:00"
fi
if [[ $num_data == 60 ]]; then
    TRAIN_TIME="02:00:00"
fi
if [[ $num_data == 120 ]]; then
    TRAIN_TIME="04:00:00"
fi
if [[ $num_data == 240 ]]; then
    TRAIN_TIME="05:30:00"
fi

############## TRAIN ################
cp scripts/train_template.sh scripts/train/${SETTING_CAPITAL}_Option=${OPTION_INDICES}_Data=${num_data}k.sh

## UPDATE TRAIN TIME / ARRAY INDEX / JOB NAME
sed -i "s#-t 02:00:00#-t ${TRAIN_TIME}#" scripts/train/${SETTING_CAPITAL}_Option=${OPTION_INDICES}_Data=${num_data}k.sh
sed -i "s#--array=#--array=${OPTION_INDICES}#" scripts/train/${SETTING_CAPITAL}_Option=${OPTION_INDICES}_Data=${num_data}k.sh
sed -i "s#--job-name=#--job-name=Train_${SETTING_CAPITAL}_Data=${num_data}k#" scripts/train/${SETTING_CAPITAL}_Option=${OPTION_INDICES}_Data=${num_data}k.sh

if [[ ${sequential} == "True" ]]; then
    echo "Running with Sequential = True"
    sed -i "s#group_by_modality_length#sequential#" scripts/train/${SETTING_CAPITAL}_Option=${OPTION_INDICES}_Data=${num_data}k.sh
fi

## LAUNCH
job_id=$(sbatch --parsable ./scripts/train/${SETTING_CAPITAL}_Option=${OPTION_INDICES}_Data=${num_data}k.sh)
echo "TRAIN JOB ID: ${job_id}"

############## EVAL ################
cp scripts/eval_template.sh scripts/eval/${SETTING_CAPITAL}_Option=${OPTION_INDICES}_Data=${num_data}k.sh

## UPDATE ARRAY INDEX / JOB NAME
sed -i "s#--array=#--array=${OPTION_INDICES}#" scripts/eval/${SETTING_CAPITAL}_Option=${OPTION_INDICES}_Data=${num_data}k.sh
sed -i "s#--job-name=#--job-name=Eval_${SETTING_CAPITAL}_Data=${num_data}k#" scripts/eval/${SETTING_CAPITAL}_Option=${OPTION_INDICES}_Data=${num_data}k.sh
sed -i "s#--dependency=""#--dependency=aftercorr:${job_id}#" scripts/eval/${SETTING_CAPITAL}_Option=${OPTION_INDICES}_Data=${num_data}k.sh

## LAUNCH
job_id=$(sbatch --parsable scripts/eval/${SETTING_CAPITAL}_Option=${OPTION_INDICES}_Data=${num_data}k.sh)
echo "EVAL JOB ID: ${job_id}"
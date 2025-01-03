# First get data for all possible grid configurations 
# In this step, we do not generate images

# SIMPLE
# number of datapoints to generate
num_data=40000
# please set the output directory where you want to store the generated files
output_dir="../../synthetic_data/grid_navigation"
dis_split_='SIMPLE'
for nrow in {8..12..1}; do #<-- number of rows in grid
for ncol in {8..12..1}; do #<-- number of columns in grid
for tot_objs in 1 2; do  #<-- total objects to spread in the grid
seed=$(( (nrow - 8) * 10 + (ncol - 8) * 2 + tot_objs ));
python create_data.py --nrow ${nrow} --ncols ${ncol} --num_data ${num_data} --seed ${seed} --tot_objs ${tot_objs} --dis_split_ ${dis_split_} --output_dir ${output_dir};
done
done
done

# HARD
dis_split_='HARD'
num_data=40000
for nrow in {8..12..1}; do #<-- number of rows in grid
for ncol in {8..12..1}; do #<-- number of columns in grid
for tot_objs in 2 3 4 5; do  #<-- total objects to spread in the grid
seed=$(( 50 + (nrow - 8) * 15 + (ncol - 8) * 3 + tot_objs ));
python create_data.py --nrow ${nrow} --ncols ${ncol} --num_data ${num_data} --seed ${seed} --tot_objs ${tot_objs} --dis_split_ ${dis_split_} --output_dir ${output_dir};
done
done
done

# Second step is to generate SIMPLE and HARD samples by rejection sampling
# SIMPLE
num_data=120000
output_dir="../../synthetic_data/grid_navigation"
dis_split_='SIMPLE'
min_steps=10
max_steps=25
python create_splits.py --num_data ${num_data} --output_dir ${output_dir} --dis_split_ ${dis_split_} --min_steps ${min_steps} --max_steps ${max_steps}

# HARD
num_data=120000
output_dir="../../synthetic_data/grid_navigation"
dis_split_='HARD'
min_steps=26
max_steps=60
python create_splits.py --num_data ${num_data} --output_dir ${output_dir} --dis_split_ ${dis_split_} --min_steps ${min_steps} --max_steps ${max_steps}


# Third step is to generate images for the created SIMPLE and HARD split
# SIMPLE
max_objs=2
n_obstacles='1'
output_dir="../../synthetic_data/grid_navigation"
dis_split_='SIMPLE'
split_for_parallel_processes_simple=1 #<-- Use this if you want to run multiple parallel jobs. 
for (( process_n=0; process_n < split_for_parallel_processes_simple; process_n++ )); do
    python create_images.py --max_objs ${max_objs} --n_obstacles ${n_obstacles} --output_dir ${output_dir} --dis_split_ ${dis_split_} --split_for_parallel_processes ${split_for_parallel_processes_simple} --process_n ${process_n};
done

# HARD
max_objs=5
n_obstacles='3:4:5'
output_dir="../../synthetic_data/grid_navigation"
dis_split_='HARD'
split_for_parallel_processes_hard=1 #<-- Use this if you want to run multiple parallel jobs across different launch scripts 
for (( process_n=0; process_n < split_for_parallel_processes_hard; process_n++ )); do
    python create_images.py --max_objs ${max_objs} --n_obstacles ${n_obstacles} --output_dir ${output_dir} --dis_split_ ${dis_split_} --split_for_parallel_processes ${split_for_parallel_processes_hard} --process_n ${process_n};
done

# Fourth step is to create final json files that contains prompt instruction to the model along with COT and the final solution
# SIMPLE_eval
output_dir="../../synthetic_data/grid_navigation"
dis_split_='SIMPLE'
num_splits=$((1 *  split_for_parallel_processes_simple)) #<-- number of obstacle configurations * split_for_parallel_processes_simple
tokenizer_path="meta-llama/Meta-Llama-3-8B-Instruct"
python create_template.py --output_dir ${output_dir} --dis_split_ ${dis_split_} --num_splits ${num_splits} --tokenizer_path ${tokenizer_path}

# HARD_eval
output_dir="../../synthetic_data/grid_navigation"
dis_split_='HARD'
num_splits=$((3 *  split_for_parallel_processes_hard))   #<-- number of obstacle configurations * split_for_parallel_processes_hard
tokenizer_path="meta-llama/Meta-Llama-3-8B-Instruct"
python create_template.py --output_dir ${output_dir} --dis_split_ ${dis_split_} --num_splits ${num_splits} --tokenizer_path ${tokenizer_path}
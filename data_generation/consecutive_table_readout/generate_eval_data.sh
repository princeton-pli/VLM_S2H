# First step, we create a set of tables with length of highlighted paths restricted in range [min_pathlen, max_pathlen]
# SIMPLE
num_data=500  #<-- this is per seed
n_seeds=1    #<-- total parallel processes 
start_seed=500

# Configuration of the table
min_rows=5
max_rows=5
min_cols=10
max_cols=10

# Configuration of path length
min_pathlen=5
max_pathlen=10

output_dir="../../synthetic_data/consecutive_table_readout/SIMPLE_eval/raw_files"
for (( seed=start_seed; seed < start_seed+n_seeds; seed++ )); do  #<-- break into multiple processes if you can do parallel launch scripts
    python create_data.py --num_data ${num_data} --seed ${seed} --min_rows ${min_rows} --max_rows ${max_rows} --min_cols ${min_cols} --max_cols ${max_cols} --min_pathlen ${min_pathlen} --max_pathlen ${max_pathlen} --output_dir ${output_dir};
done


# HARD
num_data=500  #<-- this is per seed
n_seeds=1     #<-- total parallel processes, where each process refers to a random seed
start_seed=500

# Configuration of the table
min_rows=5
max_rows=5
min_cols=10
max_cols=10

# Configuration of path length
min_pathlen=25
max_pathlen=30

output_dir="../../synthetic_data/consecutive_table_readout/HARD_eval/raw_files"
for (( seed=start_seed; seed < start_seed+n_seeds; seed++ )); do  #<-- break into multiple processes if you can do parallel launch scripts
    python create_data.py --num_data ${num_data} --seed ${seed} --min_rows ${min_rows} --max_rows ${max_rows} --min_cols ${min_cols} --max_cols ${max_cols} --min_pathlen ${min_pathlen} --max_pathlen ${max_pathlen} --output_dir ${output_dir};
done

# Second step, we create files with prompt instruction for different supervision examples
# SIMPLE
input_dir="../../synthetic_data/consecutive_table_readout/SIMPLE_eval/raw_files/Minlen5_Maxlen10"
output_dir="../../synthetic_data/consecutive_table_readout/SIMPLE_eval"
python create_splits.py --input_dir ${input_dir} --output_dir ${output_dir}


input_dir="../../synthetic_data/consecutive_table_readout/HARD_eval/raw_files/Minlen25_Maxlen30"
output_dir="../../synthetic_data/consecutive_table_readout/HARD_eval"
python create_splits.py --input_dir ${input_dir} --output_dir ${output_dir}

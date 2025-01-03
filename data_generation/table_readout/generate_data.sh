# First step, create tables with highlighted paths of different configurations
# SIMPLE
# Component 1: random paths that can have arbitrary turns but restricted by number of turns [1,2,3,4]
num_data=500  #<-- this is per seed
n_seeds=80    #<-- total parallel processes 
start_seed=0

# Configuration of the table
min_rows=7
max_rows=12
min_cols=7
max_cols=10

max_pieces=4
output_dir="../../synthetic_data/table_readout/SIMPLE/raw_files"
for (( seed=start_seed; seed < start_seed+n_seeds; seed++ )); do  #<-- break into multiple processes if you can do parallel launch scripts
    python create_random_path.py --num_data ${num_data} --seed ${seed} --min_rows ${min_rows} --max_rows ${max_rows} --min_cols ${min_cols} --max_cols ${max_cols} --output_dir ${output_dir} --max_pieces ${max_pieces};
done


# Component 2: random sinusoidal paths restricted by number of turns [3,4]
num_data=500   #<-- this is per seed
n_seeds=20     #<-- total parallel processes 
start_seed=$((seed + 1))

# Configuration of the table
min_rows=7
max_rows=12
min_cols=7
max_cols=10

output_dir="../../synthetic_data/table_readout/SIMPLE/raw_files"
for (( seed=start_seed; seed < start_seed+n_seeds; seed++ )); do  #<-- break into multiple processes if you can do parallel launch scripts
    for gap in 0 1; do
        for npieces in 3 4; do
            curr_seed=$(( start_seed + 4 * (seed - start_seed) + gap*2 + npieces-3 ));
            python create_sine_path.py --num_data ${num_data} --seed ${curr_seed} --min_rows ${min_rows} --max_rows ${max_rows} --min_cols ${min_cols} --max_cols ${max_cols} --output_dir ${output_dir} --gap ${gap} --npieces ${npieces};
        done
    done
done

# Component 2: random spiral paths restricted by number of turns [3,4]
num_data=500  #<-- this is per seed
n_seeds=20    #<-- total parallel processes 
start_seed=$((curr_seed + 1))

# Configuration of the table
min_rows=7
max_rows=12
min_cols=7
max_cols=10

max_pieces=4
output_dir="../../synthetic_data/table_readout/SIMPLE/raw_files"
for (( seed=start_seed; seed < start_seed+n_seeds; seed++ )); do  #<-- break into multiple processes if you can do parallel launch scripts
    for gap in 0 1; do
        for npieces in 3 4; do
            curr_seed=$(( start_seed + 4 * (seed - start_seed) + gap*2 + npieces-2 ));
            python create_spiral_path.py --num_data ${num_data} --seed ${curr_seed} --min_rows ${min_rows} --max_rows ${max_rows} --min_cols ${min_cols} --max_cols ${max_cols} --output_dir ${output_dir} --gap ${gap} --npieces ${npieces};
        done
    done
done



#HARD
# Component 1: longer sinusoidal paths defined by number of turns [8,10]
num_data=500  #<-- this is per seed
n_seeds=20    #<-- total parallel processes 
start_seed=1000

min_rows=7
max_rows=12
min_cols=7
max_cols=10

max_pieces=4
output_dir="../../synthetic_data/table_readout/HARD/raw_files"
for (( seed=start_seed; seed < start_seed+n_seeds; seed++ )); do  #<-- break into multiple processes if you can do parallel launch scripts
    for gap in 0 1; do
        for npieces in 8 10; do
            curr_seed=$(( start_seed + 4 * (seed - start_seed) + gap*2 + (npieces-8)/2 ));
            python create_sine_path.py --num_data ${num_data} --seed ${curr_seed} --min_rows ${min_rows} --max_rows ${max_rows} --min_cols ${min_cols} --max_cols ${max_cols} --output_dir ${output_dir} --gap ${gap} --npieces ${npieces};
        done
    done
done

# Component 2: longer spiral paths defined by number of turns [6,8]
num_data=500  #<-- this is per seed
n_seeds=20    #<-- total parallel processes 
start_seed=$((curr_seed + 1))

min_rows=7
max_rows=12
min_cols=7
max_cols=10

max_pieces=4
output_dir="../../synthetic_data/table_readout/HARD/raw_files"
for (( seed=start_seed; seed < start_seed+n_seeds; seed++ )); do  #<-- break into multiple processes if you can do parallel launch scripts
    for gap in 0 1; do
        for npieces in 6 8; do
            curr_seed=$(( start_seed + 4 * (seed - start_seed) + gap*2 + (npieces-6)/2 ));
            python create_spiral_path.py --num_data ${num_data} --seed ${curr_seed} --min_rows ${min_rows} --max_rows ${max_rows} --min_cols ${min_cols} --max_cols ${max_cols} --output_dir ${output_dir} --gap ${gap} --npieces ${npieces};
        done
    done
done

# Component 3: composition of sinusoidal and spiral paths defined by number of turns [8,10]
num_data=500  #<-- this is per seed
n_seeds=20    #<-- total parallel processes 
start_seed=$((curr_seed + 1))

min_rows=7
max_rows=12
min_cols=7
max_cols=10

max_pieces=4
output_dir="../../synthetic_data/table_readout/HARD/raw_files"
for (( seed=start_seed; seed < start_seed+n_seeds; seed++ )); do  #<-- break into multiple processes if you can do parallel launch scripts
    for gap in 0 1; do
        for npieces in 8 10; do
            curr_seed=$(( start_seed + 4 * (seed - start_seed) + gap*2 + (npieces-8)/2 ));
            python create_compose_path.py --num_data ${num_data} --seed ${curr_seed} --min_rows ${min_rows} --max_rows ${max_rows} --min_cols ${min_cols} --max_cols ${max_cols} --output_dir ${output_dir} --gap ${gap} --npieces ${npieces};
        done
    done
done


# now create files for different supervision types on SIMPLE and HARD
input_dir="../../synthetic_data/table_readout/SIMPLE/raw_files"
output_dir="../../synthetic_data/table_readout/SIMPLE/"
python create_splits.py --input_dir ${input_dir} --output_dir ${output_dir}

input_dir="../../synthetic_data/table_readout/HARD/raw_files"
output_dir="../../synthetic_data/table_readout/HARD/"
python create_splits.py --input_dir ${input_dir} --output_dir ${output_dir}

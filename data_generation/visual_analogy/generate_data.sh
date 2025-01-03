
output_dir="../../synthetic_data/visual_analogy"

# Generate puzzles
python generate_puzzles.py --simple-data 120000 --simple-eval-data 500 --hard-data 120000 --hard-eval-data 500 --output_dir ${output_dir} --nshot 2 --simple --hard

# Generate raw image / text pairs
python create_image_text_pairs.py --input-dir ${output_dir}/SIMPLE/raw_files --output-dir ${output_dir}/SIMPLE/raw_files --num-workers 32
python create_image_text_pairs.py --input-dir ${output_dir}/HARD/raw_files --output-dir ${output_dir}/HARD/raw_files --num-workers 32
python create_image_text_pairs.py --input-dir ${output_dir}/SIMPLE_eval/raw_files --output-dir ${output_dir}/SIMPLE_eval/raw_files
python create_image_text_pairs.py --input-dir ${output_dir}/HARD_eval/raw_files --output-dir ${output_dir}/HARD_eval/raw_files

# Generate Image, Text, Image-via-Text supervision data
python prepare_main_data.py --input-dir ${output_dir}/SIMPLE/raw_files --output-dir ${output_dir}/SIMPLE
python prepare_main_data.py --input-dir ${output_dir}/HARD/raw_files --output-dir ${output_dir}/HARD
python prepare_main_data.py --input-dir ${output_dir}/SIMPLE_eval/raw_files --output-dir ${output_dir}/SIMPLE_eval --eval
python prepare_main_data.py --input-dir ${output_dir}/HARD_eval/raw_files --output-dir ${output_dir}/HARD_eval --eval
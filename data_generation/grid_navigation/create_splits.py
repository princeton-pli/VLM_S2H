import json 
import os
import pandas as pd
import sys 
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_data", default=100, type=int)
    parser.add_argument("--output_dir", default=".", type=str)
    parser.add_argument("--dis_split_", default="SIMPLE", type=str)
    parser.add_argument("--min_steps", default=10, type=int)
    parser.add_argument("--max_steps", default=25, type=int)

    args = parser.parse_args()

    np.random.seed(42)

    dis_split_ = args.dis_split_
    tot_data = args.num_data
    input_dir = os.path.join(args.output_dir, dis_split_, "raw_files")
    output_dir = os.path.join(args.output_dir, dis_split_)

    json_files = [pos_json for pos_json in os.listdir(input_dir) if pos_json.endswith('.json')]
    all_data = []

    avg_steps = [] 
    for f in json_files:
        file_ = os.path.join(input_dir, f)
        data = json.load(open(file_, 'r'))
        filtered_data = []
        for d in data:
            if args.min_steps <= d['num_steps'] <= args.max_steps:
                filtered_data += [d]
        np.random.shuffle(filtered_data)
        all_data += filtered_data[:tot_data//len(json_files)]

    os.makedirs(output_dir, exist_ok=True)
    json.dump(all_data, open(os.path.join(output_dir, 'raw.json'), 'w'))



from vlmeval.smp import *
import os
import re
import csv
import json
import numpy as np
from tqdm import tqdm

### CONSECUTIVE TABLE READOUT, TABLE READOUT
def extract_list(output):
    if isinstance(output, int) or isinstance(output, float): return output
    output = re.sub(r"(\d),(\d)", r"\1\2", output)
    output = output.split('total is')[-1].split('=')[0]
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    return ','.join(numbers)

### GRID NAVIGATION
def extract_index(lst, a):
    idx = -1
    for i, il in enumerate(lst):
        if il.strip() == a.strip():
            idx = i 
    return idx

def extract_directions(output):
    try:
        dir_ = output.split('Answer: ')[-1].strip()
        ans_string = []
        all_dirs = ['left', 'right', 'up', 'down']
        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for d in dir_.split(' '):
            if d.strip != '':
                extracted_idx = extract_index(all_dirs, d.strip())
                if extracted_idx != -1:
                    ans_string += [actions[extracted_idx]]
                else:
                    ans_string = []
                    break
    except:
        return []
    return ans_string

### VISUAL ANALOGY
def extract_analogy_reasoning(output):
    if 'Example 1:' not in output: return None
    output = "Example 1: " + output.split('Example 1:')[-1].strip()
    return output

### CONSECUTIVE TABLE READOUT, TABLE READOUT
def build_table(data, index):
    new_data = []
    for n, d in tqdm(enumerate(data)):
        new_d = {}
        new_d["index"] = index + '_' + str(n)
        new_d["question"] = d["conversations"][0]["value"].replace("\n<image>", "").replace("<image>", "").strip()
        new_d["answer"] = extract_list(d["conversations"][1]["value"])
        if "image_RGB" in d.keys() and len(d["image_RGB"]) > 0:
            img_path, img_counter = d["image_RGB"]
            rgb_img = np.load(img_path)[img_counter]
            new_d["image"] = encode_image_base64(rgb_img)
        new_data.append(new_d)
    return new_data

### GRID NAVIGATION
def build_grid(data, index):
    new_data = []
    for n, d in tqdm(enumerate(data)):
        new_d = {}
        new_d["index"] = index + '_' + str(n)
        new_d["question"] = d["conversations"][0]["value"].replace("\n<image>", "").replace("<image>", "")
        new_d["num_steps"] = d["num_steps"]
        new_d["answer"] = extract_directions(d["conversations"][1]["value"])
        new_d["start_row"] = d["start_row"]
        new_d["start_col"] = d["start_col"]
        new_d["end_row"] = d["end_row"]
        new_d["end_col"] = d["end_col"]
        new_d["end_col"] = d["end_col"]
        new_d["text"] = d["text"]
        new_d["collect_objects_coordinates"] = d["collect_objects_coordinates"]
        if "image_RGB" in d.keys() and len(d["image_RGB"]) > 0:
            img_path, img_counter = d["image_RGB"]
            rgb_img = np.load(img_path)[img_counter]
            new_d["image"] = encode_image_base64(rgb_img)
        new_data.append(new_d)
    return new_data

### VISUAL ANALOGY
def build_analogy(data, index):
    new_data = []
    for n, d in tqdm(enumerate(data)):
        new_d = {}
        new_d["index"] = index + '_' + str(n)
        new_d["question"] = d["conversations"][0]["value"].replace("\n<image>", "").replace("<image>", "").strip()
        new_d["answer"] = extract_analogy_reasoning(d["conversations"][1]["value"])
        if "image_RGB" in d.keys() and len(d["image_RGB"]) > 0:
            img_path, img_counter = d["image_RGB"]
            rgb_img = np.load(img_path)[img_counter]
            new_d["image"] = encode_image_base64(rgb_img)
        new_data.append(new_d)
    return new_data

os.makedirs("dataset", exist_ok=True)

### CONSECUTIVE TABLE READOUT
for split in ["SIMPLE", "HARD"]:
    data_filename = "../synthetic_data/consecutive_table_readout/{}_eval/text2answer+cot.json".format(split)
    data = json.load(open(data_filename))
    new_data = build_table(data, "0")
    with open("dataset/OurEval_ConsecutiveTableReadout_{}_text.tsv".format(split), 'w') as f:
        dw = csv.DictWriter(f, new_data[0].keys(), delimiter='\t')
        dw.writeheader()
        dw.writerows(new_data)

    data_filename = "../synthetic_data/consecutive_table_readout/{}_eval/image2answer+text+cot.json".format(split)
    data = json.load(open(data_filename))
    new_data = build_table(data, "0")
    with open("dataset/OurEval_ConsecutiveTableReadout_{}_image.tsv".format(split), 'w') as f:
        dw = csv.DictWriter(f, new_data[0].keys(), delimiter='\t')
        dw.writeheader()
        dw.writerows(new_data)

### TABLE READOUT
for split in ["SIMPLE", "HARD"]:
    data_filename = "../synthetic_data/table_readout/{}_eval/text2answer+cot.json".format(split)
    data = json.load(open(data_filename))
    new_data = build_table(data, "0")
    with open("dataset/OurEval_TableReadout_{}_text.tsv".format(split), 'w') as f:
        dw = csv.DictWriter(f, new_data[0].keys(), delimiter='\t')
        dw.writeheader()
        dw.writerows(new_data)

    data_filename = "../synthetic_data/table_readout/{}_eval/image2answer+text+cot.json".format(split)
    data = json.load(open(data_filename))
    new_data = build_table(data, "0")
    with open("dataset/OurEval_TableReadout_{}_image.tsv".format(split), 'w') as f:
        dw = csv.DictWriter(f, new_data[0].keys(), delimiter='\t')
        dw.writeheader()
        dw.writerows(new_data)

### GRID NAVIGATION
for split in ["SIMPLE", "HARD"]:
    data_filename = "../synthetic_data/grid_navigation/{}_eval/text2answer+cot.json".format(split)
    data = json.load(open(data_filename))
    new_data = build_grid(data, "0")
    with open("dataset/OurEval_GridNavigation_{}_text.tsv".format(split), 'w') as f:
        dw = csv.DictWriter(f, new_data[0].keys(), delimiter='\t')
        dw.writeheader()
        dw.writerows(new_data)

    data_filename = "../synthetic_data/grid_navigation/{}_eval/image2answer+text+cot.json".format(split)
    data = json.load(open(data_filename))
    new_data = build_grid(data, "0")
    with open("dataset/OurEval_GridNavigation_{}_image.tsv".format(split), 'w') as f:
        dw = csv.DictWriter(f, new_data[0].keys(), delimiter='\t')
        dw.writeheader()
        dw.writerows(new_data)

### VISUAL ANALOGY
for split in ["SIMPLE", "HARD"]:
    data_filename = "../synthetic_data/visual_analogy/{}_eval/text2answer+cot.json".format(split)
    data = json.load(open(data_filename))
    new_data = build_analogy(data, "0")
    with open("dataset/OurEval_VisualAnalogy_{}_text.tsv".format(split), 'w') as f:
        dw = csv.DictWriter(f, new_data[0].keys(), delimiter='\t')
        dw.writeheader()
        dw.writerows(new_data)

    data_filename = "../synthetic_data/visual_analogy/{}_eval/image2answer+text+cot.json".format(split)
    data = json.load(open(data_filename))
    new_data = build_analogy(data, "0")
    with open("dataset/OurEval_VisualAnalogy_{}_image.tsv".format(split), 'w') as f:
        dw = csv.DictWriter(f, new_data[0].keys(), delimiter='\t')
        dw.writeheader()
        dw.writerows(new_data)
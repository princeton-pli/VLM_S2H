import json
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default=None, type=str)
parser.add_argument("--output_dir", default=None, type=str)
args = parser.parse_args()

input_dir=args.input_dir
output_dir=args.output_dir
os.makedirs(output_dir, exist_ok=True)


def text2answer(d):
    template = "\n\nThe table above shows the sales data of different products in different months. There is a path of highlighted cells (marked with a * symbol), that moves along rows of the table either left to right or right to left, starting from START_CELL and ending at END_CELL. Please return the total of all values in the highlighted cells. Provide the final answer as 'Answer: [total]'."
    contd_instr = "We enumerate the relevant row indices, column indices, row names, column names, and their corresponding values.\n\nRow Index, Column Index, Row Name, Column Name, Value\n"
    final_instr = "\n\nThe total is "


    instruction = {}
    instruction["id"] = d["id"]
    instruction["image_RGB"] = ""

    instruction["conversations"] = []
    human_instr = {}
    human_instr["from"] = "human"
    template_ = template.replace('START_CELL', '(' + d['start_product'] + ', ' + d['start_month'] + ')')
    human_instr["value"] = d["text"] + template_.replace('END_CELL', '(' + d['end_product'] + ', ' + d['end_month'] + ')')
    instruction["conversations"] += [human_instr]

    gpt_out = {}
    gpt_out["from"] = "gpt"
    gpt_out["value"] = contd_instr
    reasoning_ = ""
    all_vals = []
    for r in d["reasoning_steps"]:
        reasoning_ += r["row index"] + '\t' + r["col index"] + '\t' + r['row name'] + '\t' + r['col name'] + '\t' + r['value'] + '\n'
        all_vals += [int(r['value'])]
    reasoning_ += final_instr
    for v in all_vals[:-1]:
        reasoning_ += str(v) + ' + '
    reasoning_ += str(all_vals[-1]) + ' = '
    total_ = sum(all_vals)
    reasoning_ += str(total_) + '.\n\nAnswer: ' + str(total_)
    gpt_out["value"] += reasoning_
    instruction["conversations"] += [gpt_out]
    return instruction


json_files = [pos_json for pos_json in os.listdir(input_dir) if pos_json.endswith('.json')]


all_data = []
counter = 0
cnter = 0
    
for f in json_files:
    file_ = input_dir + "/" + f
    data = json.load(open(file_, 'r'))
    for d in data:
        counter += 1
        instruction = text2answer(d)
        all_data += [instruction]
    cnter += 1
# Open the file in write mode and dump the data
output_path=output_dir + "/text2answer+cot.json"  
with open(output_path, 'w') as file:
    json.dump(all_data, file, indent=2)

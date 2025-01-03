import json
import os
from transformers import AutoTokenizer
import argparse
import numpy as np
from symbols import obj_choices, obs_choices, obj_choices_symbols, obs_choices_symbols

tokenizer = None

image_description = "The image shows a 2 dimensional grid. "
text_description = "The table above shows a 2 dimensional grid. "

image2text_prompt = "Convert the image into a text version of a table, where cells colored in blue, green are respectively marked as S, D. Blank cells are given by O. Objects and obstacles are mentioned by their names."

grid_description = "The grid is filled up with objects, which you will be asked to recognize and collect, and obstacles, which you should avoid. Possible objects and obstacles are given as follows:\n"
grid_description += "Objects:" + ",".join(obj_choices) + "\n"
grid_description += "Obstacles:" + ",".join(obs_choices) + "\n"

add_grid_description_image = "Cells that don't contain any object or obstacle are left blank. "
add_grid_description_text = "Cells that don't contain any object or obstacle are marked by O. "

image2answer_prompt = "A traveller starts at the source (colored in blue) and wishes to go to the destination (colored in green). "
text2answer_prompt = "A traveller starts at the source (marked as S) and wishes to go to the destination (marked as D). "
common_instruction = "Your task is to give a list of actions ( left right up down) each indicating a movement by 1 cell that the traveller needs to follow. The traveller must recognize all objects in the grid, collect them but avoid obstacles in the process. Provide the final answer as 'Answer: [list of actions]'. If no path exists, output 'Answer: No path exists'."

reasoning_steps_TEMPLATE = "We follow a depth first search, always moving towards the destination.\nSource:( {}, {}) Destination:( {}, {})\n{}\n{}\n"

answer_TEMPLATE = "Answer:{}"

def get_obstacle_identity(OBSTACLES, OBSTACLE_IDX):
    return OBSTACLES[-OBSTACLE_IDX-1] # -1 correspond to IDX 0, -2 correspond to IDX 1....


def conv_to_arr(text):
    t = text.strip()
    rows = t.split('\n')
    all_ = []
    for row in rows:
        ind_row = []
        all_elems = row.split(':')
        for elem in all_elems:
            if elem != "":
                ind_row += [int(elem)]
        all_ += [ind_row]
    return np.asarray(all_)
        


def get_reasoning_steps(d, cot):
    DIRECTIONS = {
        "up": (-1, 0), 
        "down": (1, 0), 
        "left": (0, -1), 
        "right": (0, 1)
    }

    LEGACY_TO_REASONING = {
        "invalid": "outside",
        "visited": "visited",
        "blocked": "visited",
        "closed": "closed({})",
        #"future-objects": "future-collect ({})",
        "next": "okay",
    }
    
    OBJECTS = [item for item in d["objects"].split(":")]
    OBSTACLES =[item for item in d["obstacles"].split(":")]
    
    OBJECT_COORDINATES = [item.strip() for item in d["collect_objects_coordinates"].split(":")]
    OBJECT_IDX = [int(item.strip()) for item in d["collect_objects_idx"].split(":")]
    OBJECT_COLLECT = [OBJECTS[idx-1] for idx in OBJECT_IDX]

    START_OBJECT_ID = 3

    if cot:
        #table = [row.split(":")[:-1] for row in d["text"].strip().split("\n")]
        table = conv_to_arr(d["text"])

        reasoning_steps = []
        checkpoint = -1
        for step in d["reasoning_steps"]:
            # print (step)
            row, col, action = step["row"] + 1, step["col"] + 1, step["action"]
            future_checkpoint = step["checkpoint"]

            # If we pick an object, we mention here
            additional_inf = ""
            if checkpoint != -1:
                additional_inf += " Collect" + OBJECT_COLLECT[checkpoint-1]

            reasoning_step = " {}, {}:{}\n".format(chr(ord('`')+row), chr(ord('`')+col), additional_inf)

            for considered_direction in step["considered_directions"]:
                row_delta, col_delta = DIRECTIONS[considered_direction["action"]]
                new_row, new_col = row + row_delta, col + col_delta

                if considered_direction["result"] == 'closed':
                    # print (table[new_row-1][new_col-1])
                    obstacle_val = get_obstacle_identity(OBSTACLES, int(table[new_row-1][new_col-1]))
                    # print (obstacle_val)
                    reasoning_step += " {}: {}, {} {}\n".format(considered_direction["action"], chr(ord('`')+new_row), chr(ord('`')+new_col), LEGACY_TO_REASONING[considered_direction["result"]].format(obstacle_val))
                elif 'future-objects-idx-' in considered_direction["result"]:
                    future_object = OBJECT_COLLECT[int(considered_direction["result"].split("future-objects-idx-")[-1])-1]
                    legacy = 'future-collect ({})'
                    reasoning_step += " {}: {}, {} {}\n".format(considered_direction["action"], chr(ord('`')+new_row), chr(ord('`')+new_col), legacy.format(future_object))
                elif 'future-objects-destination' in considered_direction["result"]:
                    future_object = 'destination'
                    legacy = 'future-collect ({})'
                    reasoning_step += " {}: {}, {} {}\n".format(considered_direction["action"], chr(ord('`')+new_row), chr(ord('`')+new_col), legacy.format(future_object))
                else:
                    reasoning_step += " {}: {}, {} {}\n".format(considered_direction["action"], chr(ord('`')+new_row), chr(ord('`')+new_col), LEGACY_TO_REASONING[considered_direction["result"]])

            if action == "retrace":
                reasoning_step += "No available actions. retrace 1 step\n"
            else:
                row_delta, col_delta = DIRECTIONS[action]
                new_row, new_col = row + row_delta, col + col_delta  
                
                

                reasoning_step += " {}: {}, {} {}\n".format(action, chr(ord('`')+new_row), chr(ord('`')+new_col), LEGACY_TO_REASONING["next"])

            reasoning_steps.append(reasoning_step.rstrip())

            checkpoint = future_checkpoint
        # object_coordinates = d["collect_objects_coordinates"]
        object_coordinates = "Collect objects:"
        for obj_idx, obj in enumerate(OBJECT_COLLECT):
            coor = OBJECT_COORDINATES[obj_idx].split('-')
            # print (OBJECT_COORDINATES[obj_idx], coor)
            object_coordinates += obj + "( {}, {})".format(chr(ord('`')+int(coor[0])), chr(ord('`')+int(coor[1])))
        obstacle_coordinates = "\nAvoid"
        for item in OBSTACLES:
            obstacle_coordinates += item
        object_coordinates += obstacle_coordinates

        reasoning_steps = reasoning_steps_TEMPLATE.format(chr(ord('`')+d["start_row"]), chr(ord('`')+d["start_col"]), chr(ord('`')+d["end_row"]), chr(ord('`')+d["end_col"]), object_coordinates, "\n".join(reasoning_steps))
    else:
        reasoning_steps = ""
    
    return reasoning_steps

def convert_text_table(text, objects, obstacles):
    
    table = conv_to_arr(text)
    table_str = ""
    n, m = table.shape
    col_names = " &" + " &".join([" {}".format(chr(ord('`')+i)) for i in range(1, m + 1)])
    table_str = col_names+"\n"
    for i in range(n):
        row=" &"
        for j in range(m):
            if table[i, j] == 1:
                row += " S"
            elif table[i, j] == 2:
                row += " D"
            elif table[i, j] == 0:
                row += " O"
            elif table[i, j] > 2:
                row += objects[table[i, j]-3]
            elif table[i, j] < 0:
                row += obstacles[-table[i, j]-1]
            row += " &" 
        table_str += " {}{}\n".format(chr(ord('`')+i+1), row[:-2])    

    return table_str


def process_image2text(data, cot=True):
    new_data = []
    for d in data:
        prompt = "<image>\n\n{}{}".format(image_description, image2text_prompt)
        text = convert_text_table(d["text"])

        conversations = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": text}
        ]

        conversations_tokenizer = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text}
        ]

        new_d = {
            "id": d["id"],
            "image_RGB": d["image_RGB"],
            "num_tokens": len(tokenizer.apply_chat_template(conversations_tokenizer)),
            "conversations": conversations
        }
        new_data.append(new_d)
    return new_data

def process_image2answer(data, cot=True):
    new_data = []
    for d in data:
        prompt = "<image>\n\n{}{}{}{}{}".format(image_description, grid_description, add_grid_description_image, image2answer_prompt, common_instruction)
        
        all_objects_in_image = d["objects"].split(":")
        all_obstacles_in_image = d["obstacles"].split(":")
        
        
        objects_to_collect = [int(item) for item in d["collect_objects_idx"].split(":")]
        objects_to_collect = [all_objects_in_image[idx-1] for idx in objects_to_collect] 

        objects_to_collect_str = "".join(objects_to_collect)
       
        obstacles_str = "".join(all_obstacles_in_image)
        reasoning_steps = get_reasoning_steps(d, cot)

        if d["is_solvable"]:
            answer_action_steps = [" " + step for step in d["answer_steps"]]
            answer = answer_TEMPLATE.format("".join(answer_action_steps))
        else:
            answer = "Answer: No path exists"

        response = "{}{}".format(reasoning_steps, answer)

        conversations = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response}
        ]

        conversations_tokenizer = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        new_d = {
            "id": d["id"],
            "image_RGB": d["image_RGB"],
            "num_steps": d["num_steps"],
            "num_tokens": len(tokenizer.apply_chat_template(conversations_tokenizer)),
            "start_row": d["start_row"],
            "start_col": d["start_col"],
            "end_row": d["end_row"],
            "end_col": d["end_col"],
            "text": d["text"],
            "conversations": conversations,
            "obstacles": d["obstacles"],
            "objects": d["objects"],
            "collect_objects_coordinates": d["collect_objects_coordinates"],
            "collect_objects_idx": d["collect_objects_idx"]
        }
        new_data.append(new_d)
    return new_data

def process_text2answer(data, cot=True):
    new_data = []
    for d in data:
        try:
            all_objects_in_image = d["objects"].split(":")
            all_obstacles_in_image = d["obstacles"].split(":")
        except:
            print (d)
            exit (0)
        text_input = convert_text_table(d["text"], all_objects_in_image, all_obstacles_in_image)
        prompt = "{}\n\n{}{}{}{}{}".format(text_input, text_description, grid_description, add_grid_description_text, text2answer_prompt, common_instruction)
        
        
        all_objects_in_image = d["objects"].split(":")
        all_obstacles_in_image = d["obstacles"].split(":")
        
        
        objects_to_collect = [int(item) for item in d["collect_objects_idx"].split(":")]
        objects_to_collect = [all_objects_in_image[idx-1] for idx in objects_to_collect] 

        objects_to_collect_str = "".join(objects_to_collect)
        
        obstacles_str = "".join(all_obstacles_in_image)

        reasoning_steps = get_reasoning_steps(d, cot)

        if d["is_solvable"]:
            answer_action_steps = [" " + step for step in d["answer_steps"]]
            answer = answer_TEMPLATE.format("".join(answer_action_steps))
        else:
            answer = "Answer: No path exists"

        response = "{}{}".format(reasoning_steps, answer)

        conversations = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response}
        ]

        conversations_tokenizer = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        new_d = {
            "id": d["id"],
            "image_RGB": "",
            "num_steps": d["num_steps"],
            "num_tokens": len(tokenizer.apply_chat_template(conversations_tokenizer)),
            "start_row": d["start_row"],
            "start_col": d["start_col"],
            "end_row": d["end_row"],
            "end_col": d["end_col"],
            "text": d["text"],
            "conversations": conversations,
            "obstacles": d["obstacles"],
            "objects": d["objects"],
            "collect_objects_coordinates": d["collect_objects_coordinates"],
            "collect_objects_idx": d["collect_objects_idx"]
        }
        new_data.append(new_d)
    return new_data

def process_image2answer_text(data, cot=True):
    new_data = []
    for d in data:
        new_data = []
    for d in data:
        prompt = "<image>\n\n{}{}{}{}{}".format(image_description, grid_description, add_grid_description_image, image2answer_prompt, common_instruction)
        
        all_objects_in_image = d["objects"].split(":")
        all_obstacles_in_image = d["obstacles"].split(":")
        
        
        objects_to_collect = [int(item) for item in d["collect_objects_idx"].split(":")]
        objects_to_collect = [all_objects_in_image[idx-1] for idx in objects_to_collect] 

        objects_to_collect_str = "".join(objects_to_collect)
        
        obstacles_str = "".join(all_obstacles_in_image)
        reasoning_steps = get_reasoning_steps(d, cot)

        if d["is_solvable"]:
            answer_action_steps = [" " + step for step in d["answer_steps"]]
            answer = answer_TEMPLATE.format("".join(answer_action_steps))
        else:
            answer = "Answer: No path exists"

        response = "{}\n{}\n{}{}".format(image2text_prompt, convert_text_table(d["text"], all_objects_in_image, all_obstacles_in_image), reasoning_steps, answer)

        conversations = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response}
        ]

        conversations_tokenizer = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        new_d = {
            "id": d["id"],
            "image_RGB": d["image_RGB"],
            "num_steps": d["num_steps"],
            "num_tokens": len(tokenizer.apply_chat_template(conversations_tokenizer)),
            "start_row": d["start_row"],
            "start_col": d["start_col"],
            "end_row": d["end_row"],
            "end_col": d["end_col"],
            "text": d["text"],
            "conversations": conversations,
            "obstacles": d["obstacles"],
            "objects": d["objects"],
            "collect_objects_coordinates": d["collect_objects_coordinates"],
            "collect_objects_idx": d["collect_objects_idx"]
        }
        new_data.append(new_d)
    return new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".", type=str)
    parser.add_argument("--dis_split_", default="SIMPLE", type=str)
    parser.add_argument("--num_splits", default=1, type=int)
    parser.add_argument("--tokenizer_path", default='meta-llama/Meta-Llama-3-8B', type=str)

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    save_dir = os.path.join(args.output_dir, args.dis_split_)
    os.makedirs(save_dir, exist_ok=True)
    data = []
    for i in range(args.num_splits):
        with open(os.path.join(args.output_dir, args.dis_split_, "raw_files", "raw_image_{}.json".format(i))) as f:
            data += json.load(f)


    for cot, cot_tag in [(True, "+cot"), (False, "")]:
        new_data = process_text2answer(data, cot=cot)
        with open(os.path.join(save_dir, "text2answer{}.json".format(cot_tag)), "w") as f:
            json.dump(new_data, f, indent=2)

    #
    for cot, cot_tag in [(True, "+cot"), (False, "")]:

        new_data = process_image2answer(data, cot=cot)
        with open(os.path.join(save_dir, "image2answer{}.json".format(cot_tag)), "w") as f:
            json.dump(new_data, f, indent=2)

        

    for cot, cot_tag in [(True, "+cot"), (False, "")]:
        new_data = process_image2answer_text(data, cot=cot)
        with open(os.path.join(save_dir, "image2answer+text{}.json".format(cot_tag)), "w") as f:
            json.dump(new_data, f, indent=2)

        

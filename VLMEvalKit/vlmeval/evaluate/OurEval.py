from vlmeval.smp import *
import re
import ast

### CONSECUTIVE TABLE READOUT, TABLE READOUT
def extract_list(output):
    if isinstance(output, int) or isinstance(output, float): return output
    # replace numbers like `x,xxx` with `xxxx`
    output = re.sub(r"(\d),(\d)", r"\1\2", output)
    output = output.split('total is')[-1].split('=')[0]
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    return ','.join(numbers)

def extract_list_fromCoT(output):
    if isinstance(output, int) or isinstance(output, float): return output
    # replace numbers like `x,xxx` with `xxxx`
    split1 = "We enumerate the relevant row indices, column indices, row names, column names, and their corresponding values."
    split2 = "total is"
    output = output.split(split1)[-1].split(split2)[0].strip()
    all_lines = output.split('\n')
    numbers = [] 
    for line in all_lines[1:]:
        line_ = re.sub(r"(\d),(\d)", r"\1\2", line.split('\t')[-1])
        numbers += re.findall(r"[-+]?\d*\.\d+|\d+", line_)
    return ','.join(numbers)

### GRID NAVIGATION
def extract_and_convert_coordinates(input_string):
    # Define a helper function to map letters to numbers
    def letter_to_number(letter):
        return ord(letter.lower()) - ord('a') + 1

    # Extract the source and destination parts
    source_part = input_string.split("Source:")[-1].split("Destination:")[0].strip()
    destination_part = input_string.split("Destination:")[-1].strip()

    # Extract coordinates from each part
    source_coords = source_part.strip("() ").split(", ")
    destination_coords = destination_part.strip("() ").split(", ")

    # Convert letters to numbers
    try:
        source_numeric = tuple(letter_to_number(coord) for coord in source_coords)
        destination_numeric = tuple(letter_to_number(coord) for coord in destination_coords)
    except:
        source_numeric = (-1, -1)
        destination_numeric = (-1, -1)

    return source_numeric, destination_numeric

def extract_source_dest(output):
    output = 'Source:'+output.split('Source:')[-1].split('Collect objects:')[0]
    source_numeric, destination_numeric = extract_and_convert_coordinates(output)
    return int(source_numeric[0]), int(source_numeric[1]), int(destination_numeric[0]), int(destination_numeric[1])
    
def extract_index(lst, a):
    idx = -1
    for i, il in enumerate(lst):
        if il.strip() == a.strip():
            idx = i 
    return idx

def extract_directions(output):
    try:
        dir_ = output.split('Answer: ')[-1].strip()
        ans_list = []
        all_dirs = ['left', 'right', 'up', 'down']
        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for d in dir_.split(' '):
            if d.strip != '':
                extracted_idx = extract_index(all_dirs, d.strip())
                if extracted_idx != -1:
                    ans_list += [actions[extracted_idx]]
                else:
                    ans_list = []
                    break
    except:
        return []
    return ans_list


def verify_path(grid,  prediction, start_r, start_c, end_r, end_c, collect_objects_coordinates):
    object_coor = collect_objects_coordinates.split(":")
    object_coor = [s.split('-') for s in object_coor]
    object_coor = [[int(item) for item in sublist] for sublist in object_coor]
    object_checklist = [False] * len(object_coor)
    try:
        success = 0
        action_counter = 0
        curr_r, curr_c = start_r, start_c
        nrows, ncols = len(grid), len(grid[0])
        while True:
            # verify whether I visit a cell containing an object
            for obj_cnt in range(len(object_coor)):
                if (curr_r, curr_c) == (object_coor[obj_cnt][0], object_coor[obj_cnt][1]):
                    object_checklist[obj_cnt] = True
            if curr_r == end_r and curr_c == end_c:
                success = 1
                break 
            if curr_r > nrows or curr_c > ncols:
                break
            if curr_r < 0 or curr_c < 0:
                break    
            if grid[curr_r-1][curr_c-1] <= -1:
                break
            if action_counter >= len(prediction):
                break
            next_mov_r, next_mov_c = prediction[action_counter]
            action_counter += 1
            curr_r, curr_c = curr_r + next_mov_r, curr_c + next_mov_c        
        return success and all(object_checklist)
    except:
        print("FAILED ATTEMPT AT VERIFY PATH")
        return 0

def partially_verify_path(grid,  prediction, start_r, start_c, end_r, end_c, collect_objects_coordinates):
    object_coor = collect_objects_coordinates.split(":")
    object_coor = [s.split('-') for s in object_coor]
    object_coor = [[int(item) for item in sublist] for sublist in object_coor]
    object_checklist = [False] * len(object_coor)
    try:
        success = 0
        action_counter = 0
        curr_r, curr_c = start_r, start_c
        nrows, ncols = len(grid), len(grid[0])
        num_obstacles = 0
        while True:
            # verify whether I visit a cell containing an object
            for obj_cnt in range(len(object_coor)):
                if (curr_r, curr_c) == (object_coor[obj_cnt][0], object_coor[obj_cnt][1]):
                    object_checklist[obj_cnt] = True
            if curr_r == end_r and curr_c == end_c:
                success = 1
                break 
            if curr_r > nrows or curr_c > ncols:
                break
            if curr_r < 0 or curr_c < 0:
                break    
            if grid[curr_r-1][curr_c-1] <= -1:
                num_obstacles += 1
            if action_counter >= len(prediction):
                break
            next_mov_r, next_mov_c = prediction[action_counter]
            action_counter += 1
            curr_r, curr_c = curr_r + next_mov_r, curr_c + next_mov_c
        return success, num_obstacles, sum(object_checklist) 
    except:
        print("FAILED ATTEMPT AT VERIFY PATH")
        return 0, None

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
        
def measure_accuracy(grid, prediction, start_r, start_c, end_r, end_c, collect_objects_coordinates):
    grid_arr = conv_to_arr(grid)
    return verify_path(grid_arr,  prediction, start_r, start_c, end_r, end_c, collect_objects_coordinates)

def partially_measure_accuracy(grid, prediction, start_r, start_c, end_r, end_c, collect_objects_coordinates):
    #first convert grid, which is in string format, to an array
    grid_arr = conv_to_arr(grid)
    return partially_verify_path(grid_arr,  prediction, start_r, start_c, end_r, end_c, collect_objects_coordinates)

### VISUAL ANALOGY
def extract_analogy_reasoning(output):
    if 'Example 1:' not in output: return None
    output = "Example 1: " + output.split('Example 1:')[-1].strip()
    return output

def extract_analogy_answer(output):
    output = output.strip()
    try:
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        answer_option = int(numbers[-1])
        lines = output.split("\n")
        for i in range(2, 6):
            if i == 6 - answer_option:
                assert "consistent" in lines[-i]
                assert not "not consistent" in lines[-i]
            else:
                assert "not consistent" in lines[-i]
    except:
        return -1
    return answer_option

### CONSECUTIVE TABLE READOUT, TABLE READOUT
def OurEval_Table_eval(eval_file):
    logger = get_logger('Evaluation')
    data = load(eval_file)
    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]

    num_correct = 0
    for line in lines:
        prediction = extract_list(line["prediction"])
        if prediction is None: continue
        target = line["answer"]
        if "total is" in target or "=" in target:
            target = extract_list(target)
        if target.strip() == prediction.strip():
            num_correct += 1
    em_score = num_correct / lt
    score_dict = {"exact_match" : em_score}

    if not "noCoT" in eval_file:
        num_correct = 0
        for line in lines:
            prediction = extract_list_fromCoT(line["prediction"])
            if prediction is None: continue
            target = line["answer"]
            if "total is" in target or "=" in target:
                target = extract_list_fromCoT(target)
            if target.strip() == prediction.strip():
                num_correct += 1
        em_score = num_correct / lt
        score_dict["cot_exact_match"] = em_score

    score_pth = eval_file.replace('.xlsx', '_score.json')
    dump(score_dict, score_pth)
    logger.info(f'OurEval successfully finished evaluating {eval_file}, results saved in {score_pth}')
    logger.info('Exact Match: ')
    for key, value in score_dict.items():
        logger.info('{}:{}'.format(key, value))


### GRID NAVIGATION
def OurEval_Grid_eval(eval_file):
    logger = get_logger('Evaluation')
    data = load(eval_file)
    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]
    is_correct = []
    
    num_source_dest_correct = 0
    num_partially_correct = 0
    num_correct = 0
    num_obstacles = 0
    num_obstacles_parsed = 0
    num_objects_collected = 0
    fraction_objects_collected = 0.
    total_objects = 0

    for line in lines:
        prediction = line["prediction"]
        pred_start_r, pred_start_c, pred_end_r, pred_end_c =  extract_source_dest(prediction)
        prediction = extract_directions(prediction)
        if prediction is None: continue
        target = line["answer"]
        if "Answer: " in target:
            target = extract_directions(target)
        else:
            target = ast.literal_eval(target)
        
        grid = line['text']
        start_r = int(line["start_row"])
        start_c = int(line["start_col"])
        end_r = int(line["end_row"])
        end_c = int(line["end_col"])

        collect_objects_coordinates = line['collect_objects_coordinates']
        total_objects_to_collect = len(collect_objects_coordinates.split(":"))
        total_objects += total_objects_to_collect

        is_correct.append(False)

        if pred_start_r == start_r and pred_start_c == start_c and pred_end_r == end_r and pred_end_c == end_c:
            num_source_dest_correct += 1
        if len(prediction) == 0 and len(target) == 0:
            num_correct += 1
            num_partially_correct += 1
            is_correct[-1] = True

        elif len(prediction) != 0:
            if target == prediction:
                num_correct += 1
                num_partially_correct += 1
                is_correct[-1] = True
                num_objects_collected += total_objects_to_collect
                fraction_objects_collected += 1.
            else:
                num_correct_ = measure_accuracy(grid, prediction, start_r, start_c, end_r, end_c, collect_objects_coordinates)
                num_partially_correct_, num_obstacles_, num_objects_ = partially_measure_accuracy(grid, prediction, start_r, start_c, end_r, end_c, collect_objects_coordinates)
                
                num_obstacles_parsed += 1
                num_obstacles += num_obstacles_
                num_partially_correct += num_partially_correct_
                num_correct += num_correct_
                is_correct[-1] = num_correct_ == 1
                num_objects_collected += num_objects_
                fraction_objects_collected += (num_objects_/(1.*total_objects_to_collect))

    em_score = num_correct / lt
    partial_score = num_partially_correct / lt
    source_dest_score = num_source_dest_correct / lt
    fraction_objects_collected_score = fraction_objects_collected/lt
    if num_obstacles_parsed > 0:
        avg_num_obstacles = num_obstacles / num_obstacles_parsed
    else:
        avg_num_obstacles = -1

    all_fraction_objects_collected_score = (num_objects_collected/(1.*total_objects))

    score_dict = {"exact_match" : em_score, \
        "partial_score": partial_score, \
        "source_dest_score": source_dest_score, \
        "avg_num_obstacles": avg_num_obstacles, \
        "is_correct": is_correct, \
        "fraction_objects_collected_score": fraction_objects_collected_score, \
        "all_fraction_objects_collected_score": all_fraction_objects_collected_score
    }
    
    logger.info('Exact Match: ')
    for key, value in score_dict.items():
        logger.info('{}:{}'.format(key, value))
    score_pth = eval_file.replace('.xlsx', '_score.json')
    dump(score_dict, score_pth)
    logger.info(f'OurEval successfully finished evaluating {eval_file}, results saved in {score_pth}')

### VISUAL ANALOGY
def OurEval_Analogy_eval(eval_file):
    logger = get_logger('Evaluation')
    data = load(eval_file)
    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]
    num_fully_correct = 0
    num_partially_correct = 0
    for line in lines:
        prediction = extract_analogy_reasoning(line["prediction"])
        if prediction is None: continue
        target = line["answer"]
        if target.strip() == prediction.strip():
            num_fully_correct += 1
        if extract_analogy_answer(target) == extract_analogy_answer(prediction):
            num_partially_correct += 1
    
    em_score = num_fully_correct / lt
    partial_score = num_partially_correct / lt
    score_dict = {"exact_match" : em_score, "partial_score" : partial_score}
    score_pth = eval_file.replace('.xlsx', '_score.json')
    dump(score_dict, score_pth)
    logger.info(f'OurEval successfully finished evaluating {eval_file}, results saved in {score_pth}')
    logger.info('Exact Match: ')
    for key, value in score_dict.items():
        logger.info('{}:{}'.format(key, value))
    

def parse_args():
    parser = argparse.ArgumentParser(description='Inference LLM Answers. ')
    parser.add_argument('--data', type=str, help='The xlsx file which contains the predictions of a model and the ground truth answers.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    load_env()
    args = parse_args()
    acc = OurEval_eval(eval_file=args.data)

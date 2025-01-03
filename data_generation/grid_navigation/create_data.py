import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import warnings
from matplotlib import MatplotlibDeprecationWarning
# Suppress MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def l1_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def l2_distance(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def reset(visited, new_r, new_c):
    visited = np.zeros_like(visited, dtype=bool)
    return visited

def dfs(grid, start_r, start_c, end_r, end_c, additional_objects, additional_object_val, object_counter, full_path, actions, visited):
    
    if object_counter >= len(additional_objects):
        if (start_r, start_c) == (end_r, end_c):
            return full_path, actions  # Found the path, return both the path and the full path

    visited[start_r][start_c] = True

    # Define the directions for DFS: up, down, left, right
    DIRECTIONS = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]

    # Sort directions based on Manhattan distance to prioritize the direction toward value 2
    # Move towards objects first
    if object_counter < len(additional_objects):
        obj_r, obj_c = additional_objects[object_counter]
        directions = sorted(DIRECTIONS, key=lambda d: (l1_distance(start_r + d[0], start_c + d[1], obj_r, obj_c), l2_distance(start_r + d[0], start_c + d[1], obj_r, obj_c)))
    else:
        directions = sorted(DIRECTIONS, key=lambda d: (l1_distance(start_r + d[0], start_c + d[1], end_r, end_c), l2_distance(start_r + d[0], start_c + d[1], end_r, end_c)))

    considered_directions = []
    # print (start_r, start_c, directions, object_counter)
    for dr, dc, action in directions:
        new_r, new_c = start_r + dr, start_c + dc

       
        if not (0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]):
            result = {
                "action": action,
                "result": "invalid"
            }
            considered_directions.append(result)

        elif visited[new_r][new_c]:
            result = {
                "action": action,
                "result": "visited"
            }
            considered_directions.append(result)

        elif grid[new_r][new_c] == -1:
            result = {
                "action": action,
                "result": "closed"
            }
            considered_directions.append(result)
        else:
            future_obj_visit = False
            if object_counter < len(additional_objects):
                if any([grid[new_r][new_c] == additional_object_val[o] for o in range(object_counter+1, len(additional_objects))]) :  #<-- can't visit an object earlier
                    result = {
                        "action": action,
                        "result": "future-objects-idx-" + str(object_counter+1)
                    }
                    future_obj_visit = True
                elif (new_r, new_c) == (end_r, end_c):
                    result = {
                            "action": action,
                            "result": "future-objects-destination"
                    }
                    future_obj_visit = True
            if not future_obj_visit:
                if object_counter < len(additional_objects):
                    x, y = additional_objects[object_counter]
                    if (new_r, new_c) == (x, y): 
                        object_counter += 1
                        checkpoint = object_counter
                        visited = reset(visited, new_r, new_c)
                    else:
                        checkpoint = -1
                else:
                    checkpoint = -1
                state = {
                    "row": start_r,
                    "col": start_c,
                    "action": action,
                    "checkpoint": checkpoint, 
                    "considered_directions": considered_directions
                }
                full_path.append(state)
                actions.append(action)


                full_path, final_actions = dfs(grid, new_r, new_c, end_r, end_c, additional_objects, additional_object_val, object_counter, full_path, actions, visited)

                if final_actions:  # If a valid path is found, return it
                    return full_path, final_actions
                
                actions.pop()

                result = {
                    "action": action,
                    "result": "blocked"
                }
            ## CREATE A COPY SO THAT WE DON'T APPEND TO A PREVIOUS ENTRY
            new_considered_directions = [considered_direction for considered_direction in considered_directions]
            new_considered_directions.append(result)
            considered_directions = new_considered_directions

    state = {
        "row": start_r,
        "col": start_c,
        "action": "retrace",
        "considered_directions": considered_directions,
        "checkpoint": -1, 
    }
    full_path.append(state)

    # No valid move found, return empty result
    return full_path, None
    


def fill_random_cells_with_negative_one(arr, og_start_r, og_start_c, og_end_r, og_end_c, inside_prob, outside_prob):
    # Ensure start_r, start_c is the top-left and end_r, end_c is the bottom-right
    if og_start_r > og_end_r:
        start_r, end_r = og_end_r, og_start_r
    else:
        start_r, end_r = og_start_r, og_end_r
    if og_start_c > og_end_c:
        start_c, end_c = og_end_c, og_start_c
    else:
        start_c, end_c = og_start_c, og_end_c
    
    rows, cols = arr.shape

    # Get all the cells inside the rectangle excluding the two corner cells
    inside_cells = [(r, c) for r in range(start_r, end_r+1) for c in range(start_c, end_c+1)]
    inside_cells.remove((og_start_r, og_start_c))
    inside_cells.remove((og_end_r, og_end_c))

    # Randomly select 50% of the inside cells to fill with -1
    num_to_fill_inside = int(len(inside_cells) * inside_prob)

    indices = np.random.choice(len(inside_cells), num_to_fill_inside, replace=False)
    cells_to_fill_inside = [inside_cells[k] for k in indices]
    
    # Fill the selected inside cells with -1
    for r, c in cells_to_fill_inside:
        arr[r][c] = -1


    # Get all the cells outside the rectangle
    outside_cells = [(r, c) for r in range(rows) for c in range(cols)
                     if not (start_r <= r <= end_r and start_c <= c <= end_c)]
    
    # Randomly fill cells outside the rectangle with -1 with a probability of 0.3
    for r, c in outside_cells:
        if np.random.random_sample() < outside_prob:
            arr[r][c] = -1
    return arr


def greedy_sort_indices(coordinates, start_r, start_c):
    if not coordinates:
        return []

    # Track indices and coordinates
    indices = list(range(len(coordinates)))
    sorted_indices = []
    sorted_coordinates = [(start_r, start_c)]

    while coordinates:
        # Find the closest coordinate to the last one in sorted_coordinates
        last_coord = sorted_coordinates[-1]
        closest_index = min(range(len(coordinates)), key=lambda i: l1_distance(last_coord[0], last_coord[1], coordinates[i][0], coordinates[i][1]))
        sorted_coordinates.append(coordinates.pop(closest_index))
        sorted_indices.append(indices.pop(closest_index))

    return sorted_indices

# Function to sort another list based on sorted indices
def sort_list_by_indices(another_list, sorted_indices):
    return [another_list[i] for i in sorted_indices]


def fill_objects(grid, start_r, start_c, end_r, end_c, tot_objs):

    additional_objects, additional_object_val = [], []
    n, m = grid.shape

    # first select coordinates for each object
    obj_row_cnter = np.random.choice(range(n), tot_objs, replace=True)
    obj_col_cnter = np.random.choice(range(m), tot_objs, replace=True)

    # go through the pairs and reselect if they intersect with the start and the destination
    for counter in range(tot_objs):
        obj_row, obj_col = obj_row_cnter[counter], obj_col_cnter[counter]
        while (obj_row, obj_col) == (start_r, start_c) or (obj_row, obj_col) == (end_r, end_c):
            obj_row_cnter[counter] = np.random.choice(range(n), 1, replace=True)[0]
            obj_col_cnter[counter] = np.random.choice(range(m), 1, replace=True)[0]
            obj_row, obj_col = obj_row_cnter[counter], obj_col_cnter[counter]

    for i_num in range(tot_objs):
        additional_objects += [[obj_row_cnter[i_num], obj_col_cnter[i_num]]]

    # then assign their values depending on distance to source
    sorted_indices = greedy_sort_indices(additional_objects[:], start_r, start_c)
    additional_objects = sort_list_by_indices(additional_objects, sorted_indices)
    additional_object_val = []
    for i_num in range(len(additional_objects)):
        additional_object_val += [3+i_num]


    obj_select = range(len(additional_objects))
    keep_additional_object_val = []
    keep_additional_objects = []
    for i_num, obj in enumerate(additional_objects):
        if i_num in obj_select:
            keep_additional_objects += [obj]
            keep_additional_object_val += [additional_object_val[i_num]] 
        grid[obj[0], obj[1]] = additional_object_val[i_num]

    #simple assign
    return keep_additional_objects, keep_additional_object_val
    

def create_table(nrow, ncol, tot_objs):
    arr = np.zeros((nrow, ncol), dtype=int)

    min_path_len = np.random.choice(min_path_lens, 1, replace=False)[0]
    
    inside_prob = args.inside_prob
    outside_prob = args.outside_prob

    num_tries = 0
    while True:
        
        candidates = [i for i in range(0, nrow-min_path_len, 1)] + [i for i in range(min_path_len, nrow, 1)]
        start_r = np.random.choice(list(set(candidates)), 1, replace=False)[0]
        candidates = [i for i in range(start_r+min_path_len, nrow, 1)] + [i for i in range(start_r-min_path_len, 0, -1)]
        if len(candidates) == 0:
            num_tries += 1
            continue
        end_r = np.random.choice(list(set(candidates)), 1, replace=False)[0]


        candidates = [i for i in range(0, ncol-min_path_len, 1)] + [i for i in range(min_path_len, ncol, 1)]
        start_c = np.random.choice(list(set(candidates)), 1, replace=False)[0]
        candidates = [i for i in range(start_c+min_path_len, ncol, 1)] + [i for i in range(start_c-min_path_len, 0, -1)]
        if len(candidates) == 0:
            num_tries += 1
            continue
        end_c = np.random.choice(list(set(candidates)), 1, replace=False)[0]


        num_tries += 1
        # CHECK POSSIBLE PATH LENGTH
        if np.absolute(start_r - end_r) >= min_path_len \
            and np.absolute(start_c - end_c) >= min_path_len :
            break

        if num_tries >= 10:
            option = np.random.choice(4, 1, replace=False)[0]
            start_r, end_r = (nrow - 1 + (option%2)) % nrow, (nrow - 1 + ((option+1)%2)) % nrow
            start_c, end_c = (ncol - 1 + (option//2)) % ncol, (ncol - 1 + (option//2+1)%2) % ncol                
            break
    
    start_r = int(start_r)
    start_c = int(start_c)
    end_r = int(end_r)
    end_c = int(end_c)
    
    arr[start_r, start_c] = 1 #
    arr[end_r, end_c] = 2
    #Now, fill in the rectangle between (start_r, start_c) and (end_r, end_c) with obstacles
    arr = fill_random_cells_with_negative_one(arr, start_r, start_c, end_r, end_c, inside_prob, outside_prob)

    # fill arbitrary objects, with one inside the rectangle containing the start and destination points
    additional_objects, additional_object_val = fill_objects(arr, start_r, start_c, end_r, end_c, tot_objs)
    object_counter = 0
    
    full_path, final_actions = dfs(arr, start_r, start_c, end_r, end_c, additional_objects, additional_object_val, object_counter, [], [], np.zeros_like(arr, dtype=bool))
    return arr, start_r, start_c, end_r, end_c, full_path, final_actions, additional_objects, additional_object_val


def convert_arr_to_string(arr):
    string=''
    for row in arr:
        for elem in row:
            string += ':' + str(elem)
        string += '\n'
    return string

def main(args):
    all_data = []
    images = []
    image_counter = 0
    split_counter = 0


    avg_moves = []
    no_answer = 0
    for i in tqdm(range(args.num_data)):
        
        num_rows = args.nrows
        num_cols = args.ncols

        arr, start_r, start_c, end_r, end_c, full_path, final_actions, additional_objects, additional_object_val = create_table(num_rows, num_cols, args.tot_objs)
        
        if final_actions is None: continue
        
        avg_moves += [len(full_path)]
        no_answer += final_actions is None

        data_ = {
            "id": i,
            "start_row": int(start_r) + 1,
            "start_col": int(start_c) + 1,
            "end_row": int(end_r) + 1,
            "end_col": int(end_c) + 1,
            "num_steps": len(full_path),
            "reasoning_steps": full_path,
            "is_solvable": not final_actions is None,
            "answer_steps": final_actions,
            "text": convert_arr_to_string(arr),
            "collect_objects_coordinates": ":".join(["-".join([str(k_+1) for k_ in obj_coor]) for obj_coor in additional_objects]),
            "collect_objects_idx": ":".join([str(k_-2) for k_ in additional_object_val]) #<-- index (in 1-index) of the object to collect
        }

        all_data.append(data_)


    with open(os.path.join(args.output_dir, args.dis_split_, "raw_files", 'nrows={}_ncols={}_inside={:.2f}_outside={:.2f}_seed={}_totobjs={}.json'.format(args.nrows, args.ncols, args.inside_prob, args.outside_prob, args.seed,  args.tot_objs)), "w") as f:
        json.dump(all_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_data", default=100, type=int)
    parser.add_argument("--nrows", default=7, type=int)
    parser.add_argument("--ncols", default=7, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--output_dir", default=".", type=str)
    parser.add_argument("--dis_split_", default="SIMPLE", type=str)
    parser.add_argument("--prob", default=0.35, type=float)
    parser.add_argument("--tot_objs", default=5, type=int)


    args = parser.parse_args()

    args.outside_prob = args.inside_prob = args.prob

    os.makedirs(os.path.join(args.output_dir, args.dis_split_, "raw_files"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.dis_split_, "raw_files", "images"), exist_ok=True)

    np.random.seed(args.seed)

    min_path_lens = [1]
    main(args)

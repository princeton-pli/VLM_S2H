import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
# 
from matplotlib import rcParams
import warnings
from matplotlib import MatplotlibDeprecationWarning
from PIL import Image
import sys
from symbols import obj_choices, obs_choices, obj_choices_symbols, obs_choices_symbols


# Suppress MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


# Function to create the image from an array and table
def create_image(ax, arr, darken_border):
    nrows, ncols = len(arr), len(arr[0])  # Get number of rows and columns in the array
    
    # Iterate over the array and create a big dot in each cell
    for i in range(nrows):
        for j in range(ncols):
            
            cell_value = arr[i][j]
            
            if cell_value == "1":
                cell_color = 'skyblue'
            elif cell_value == "2":
                cell_color = 'darkgreen'
            else:
                cell_color = 'white'

           # Draw the cell with the background color and add a black border to each cell
            ax.add_patch(plt.Rectangle((j, nrows-i-1), 1, 1, fill=True, facecolor=cell_color, edgecolor='black', linewidth=2))

            if not(cell_value == "" or cell_value == "1" or cell_value == "2"):
                ax.text(j + 0.5, nrows-i-0.5, arr[i][j], ha='center', va='center', fontsize=40)  # Adjust font size as needed
            
    
    # Set grid limits and enable visible axes
    # Set grid limits
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_aspect('equal')
    
    # Set 1-based row and column indices
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    
    ax.set_xticklabels([chr(ord('a') + num) for num in range(ncols)])
    ax.set_yticklabels(np.array([chr(ord('a') + num) for num in range(nrows)])[::-1])
    
    # Set tick label position (outside the grid)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True, labelsize=20)  # Adjust label font size
   
    # Enable gridlines
    ax.grid(False)
    
    # Show the borders with axes on
    ax.axis('on')  # Show the axis and indices


# Function to save the grid as an image
def create_image_RGB(arr, filename='test.png'):
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(10, 10))

    # Call create_image function to plot the table
    create_image(ax, arr, darken_border=True)
    nrows, ncols = len(arr), len(arr[0]) 
    
    
    # Save the image as a PNG file
    plt.draw()

    # Convert figure to RGB array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((height, width, 3))

    # Close the figure to free up memory
    plt.close(fig)

    return image




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
        

def convert_arr_to_string(arr):
    string=''
    for row in arr:
        for elem in row:
            string += ':' + str(elem)
        string += '\n'
    return string



def map_to_emoji(arr,obs_composition,tot_objs):
    emoji_arr = []
    map_to_emoji = {}
    map_to_emoji[1] = "1"
    map_to_emoji[2] = "2"

    # randomly select one obstacle emoji to represent obstacle
    obs_choice = np.random.choice(len(obs_choices), obs_composition, replace=False)
    for ob in range(1, obs_composition+1):
        map_to_emoji[-ob] = obs_choices_symbols[obs_choice[ob-1]]
    if obs_composition != 1:
        obs_arr = np.random.choice(range(1, obs_composition+1), arr.shape, replace=True)
        n, m = arr.shape
        for i in range(n): 
            for j in range(m):
                if arr[i, j] == -1: arr[i, j] = -obs_arr[i, j]

    # randomly select tot_objs objects
    obj_choice_ = np.random.choice(len(obj_choices), tot_objs, replace=False)
    for i in range(tot_objs):
        map_to_emoji[i+3] = obj_choices_symbols[obj_choice_[i]]

    for i in range(len(arr)):
        emoji_ = []
        for j in range(len(arr[0])):
            if arr[i, j] != 0:
                emoji_ += [map_to_emoji[arr[i, j]]]
            else:
                emoji_ += [""]
        emoji_arr += [emoji_]
    image = create_image_RGB (emoji_arr)
    
    obstacles = ":".join(obs_choices[ob] for ob in obs_choice)
    objects = ":".join([obj_choices[i] for i in obj_choice_])
    return image, obstacles, objects, convert_arr_to_string(arr)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".", type=str)
    parser.add_argument("--dis_split_", default="SIMPLE", type=str)
    parser.add_argument("--n_obstacles", default='1:2', type=str)
    parser.add_argument("--max_objs", default=1, type=int)
    parser.add_argument("--split_for_parallel_processes", default=1, type=int)
    parser.add_argument("--process_n", default=0, type=int)
    args = parser.parse_args()

    dis_split_ = args.dis_split_
    input_dir = os.path.join(args.output_dir, dis_split_)
    output_dir = os.path.join(args.output_dir, dis_split_, "raw_files")

    all_data = json.load(open(os.path.join(input_dir, 'raw.json'), 'r'))
    np.random.seed(42) 
    np.random.shuffle(all_data)

    image_counter = 0
    images = []
    split_counter = 0
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # we create k splits depending on the number of configurations we create for obstacles
    n_obstacles = [int(k) for k in args.n_obstacles.split(':')]
    num_obstacle_splits = len(n_obstacles)

    if args.split_for_parallel_processes > 1:
        left_ = args.process_n * (len(all_data) // args.split_for_parallel_processes)
        right_ = left_ + (len(all_data) // args.split_for_parallel_processes)
        all_data = all_data[left_: right_]
    
    start_index = args.process_n * len(n_obstacles)
    for data_split_, n_obstacle in enumerate(n_obstacles):
        filename = os.path.abspath(os.path.join(output_dir, "images", 'split={}_datasplit_{}.npy'.format(split_counter, start_index + data_split_)))
        new_data = []
        left_ = data_split_ * (len(all_data) // num_obstacle_splits)
        right_ = left_ + (len(all_data) // num_obstacle_splits)
        for d in tqdm(all_data[left_: right_]):
            grid_text = d['text']
            arr = conv_to_arr(grid_text)
            image, obstacles, objects, new_arr = map_to_emoji(arr, n_obstacle, args.max_objs)
            images += [image]

            d['image_RGB'] = [filename, image_counter]
            d['obstacles'] = obstacles
            d['objects'] = objects
            d['text'] = new_arr

            new_data += [d]
            image_counter += 1

            if image_counter % 100 == 0:
                images_array = np.array(images)  # Converts the list to a single 4D numpy array
                np.save(filename, images_array)
                
                split_counter += 1
                filename = os.path.abspath(os.path.join(output_dir, "images", 'split={}_datasplit_{}.npy'.format(split_counter, start_index + data_split_)))
                
                image_counter = 0
                images = []

        if len(images) > 0:
            images_array = np.array(images)  # Converts the list to a single 4D numpy array
            np.save(filename, images_array)

        json.dump(new_data, open(os.path.join(output_dir, 'raw_image_{}.json'.format(start_index + data_split_)), 'w'), indent=4)
        


            
            


import os
import sys
import pandas as pd
import numpy as np
import random
from names import products, months
import random
import matplotlib.pyplot as plt
from PIL import Image

import argparse
from tqdm import tqdm
import json


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--num_data", default=1, type=int)

parser.add_argument("--min_rows", default=5, type=int)
parser.add_argument("--min_cols", default=5, type=int)
parser.add_argument("--max_rows", default=10, type=int)
parser.add_argument("--max_cols", default=10, type=int)
parser.add_argument("--max_pieces", default=4, type=int)

parser.add_argument("--output_dir", default=".", type=str)
args = parser.parse_args()

initial_seed = args.seed
seed = initial_seed
args.output_dir += "/random_"+str(args.max_pieces)
os.makedirs(args.output_dir, exist_ok=True)


def zigzag_path(df, p=0.5, min_path_len=4):
    # Get random row and column indices for two cells
    num_tries = 0
    piece_options = np.arange(1, args.max_pieces+1)   #<-- [1,2,3,4]
    num_pieces = np.random.choice(piece_options, 1)[0]

    while (True):
        if num_pieces == 1:
            option = np.random.choice(4, 1, replace=False)[0]
            if option % 2 == 0:
                start_r, end_r = np.random.choice(len(df), 2, replace=False)
                start_c = np.random.choice(len(df.columns), 1, replace=False)[0]
                end_c = start_c
                num_tries += 1
                if np.absolute(start_r - end_r) >= min_path_len :
                    break
            else:
                start_c, end_c = np.random.choice(len(df.columns), 2, replace=False)
                start_r = np.random.choice(len(df), 1, replace=False)[0]
                end_r = start_r
                num_tries += 1
                if np.absolute(start_c - end_c) >= min_path_len :
                    break
            if num_tries >= 100:
                option = np.random.choice(4, 1, replace=False)[0]
                start_r, end_r = (len(df) - (option//2)) % len(df), (len(df) - (option//2 + option%2)%2) % len(df)
                start_c, end_c = (len(df.columns) - (option//2 + option%2)%2) % len(df.columns), (len(df.columns) - 1 + (option//2)) % len(df.columns)

                break
        else:
            candidates = [i for i in range(0, len(df)-min_path_len, 1)] + [i for i in range(min_path_len, len(df), 1)]
            start_r = np.random.choice(candidates, 1, replace=False)[0]
            candidates = [i for i in range(start_r+min_path_len, len(df), 1)] + [i for i in range(start_r-min_path_len, 0, -1)]
            if len(candidates) == 0:
                num_tries += 1
                continue
            end_r = np.random.choice(candidates, 1, replace=False)[0]


            candidates = [i for i in range(0, len(df.columns)-min_path_len, 1)] + [i for i in range(min_path_len, len(df.columns), 1)]
            start_c = np.random.choice(candidates, 1, replace=False)[0]
            candidates = [i for i in range(start_c+min_path_len, len(df.columns), 1)] + [i for i in range(start_c-min_path_len, 0, -1)]
            if len(candidates) == 0:
                num_tries += 1
                continue
            end_c = np.random.choice(candidates, 1, replace=False)[0]
            num_tries += 1
            # CHECK POSSIBLE PATH LENGTH
            if np.absolute(start_r - end_r)  >= min_path_len and np.absolute(start_c - end_c) >= min_path_len:  #<-- an arbitrary threshold on path length to make sure trasvering the path is difficult
                break

            if num_tries >= 100:
                option = np.random.choice(4, 1, replace=False)[0]
                start_r, end_r = (len(df) - 1 + option%2) % len(df), (len(df) - 1 + (option+1)%2) % len(df)
                start_c, end_c = (len(df.columns) - 1 + (option//2)) % len(df.columns), (len(df.columns) - 1 + (option//2+1)%2) % len(df.columns)   
                break

    curr_r, curr_c = start_r, start_c
    prev_r, prev_c = curr_r, curr_c

    path_info = []
    
    

    def move_right(prev_row, prev_col, curr_row, curr_col, randomize=True):
        n_rows, n_cols = len(df), len(df.columns)
        prev_row, prev_col = curr_row, curr_col
        if randomize:
            if np.sign(prev_col - end_c) > 0:
                
                cand = []
                for c in range(prev_col):
                    if np.absolute(c - end_c) > 1:
                        cand += [c]
                curr_col = np.random.choice(cand, 1)[0]
                    
            else:
                
                cand = []
                for c in range(prev_col+1, n_cols):
                    if np.absolute(c - end_c) > 1:
                        cand += [c]
                curr_col = np.random.choice(cand, 1)[0]
            
        else:
            curr_col = end_c
        return prev_row, prev_col, curr_row, curr_col
        
    def move_down(prev_row, prev_col, curr_row, curr_col, randomize=True):
        n_rows, n_cols = len(df), len(df.columns)
        prev_row, prev_col = curr_row, curr_col
        if randomize:
            if np.sign(prev_row - end_r) > 0:
                cand = []
                for r in range(prev_row):
                    if np.absolute(r - end_r) > 1:
                        cand += [r]
                
                curr_row = np.random.choice(cand, 1)[0]
            
                
            else:
                cand = []
                for r in range(prev_row+1, n_rows):
                    if np.absolute(r - end_r) > 1:
                        cand += [r]
                curr_row = np.random.choice(cand, 1)[0]
                
        else:
            curr_row = end_r
        return prev_row, prev_col, curr_row, curr_col
        
    first_move = np.random.random_sample() >= p
    move = 0
    for n in range(num_pieces):
        randomize = (num_pieces - n) > 2
        if n == 0:
            if num_pieces == 1:
                if prev_r == end_r:
                    prev_r, prev_c, curr_r, curr_c = move_right(prev_r, prev_c, curr_r, curr_c, randomize=randomize)
                    move = 1
                else:
                    prev_r, prev_c, curr_r, curr_c = move_down(prev_r, prev_c, curr_r, curr_c, randomize=randomize)
                    move = 0
            else:
                if first_move:
                    prev_r, prev_c, curr_r, curr_c = move_right(prev_r, prev_c, curr_r, curr_c, randomize=randomize)
                    move = 1
                else:
                    prev_r, prev_c, curr_r, curr_c = move_down(prev_r, prev_c, curr_r, curr_c, randomize=randomize)
                    move = 0
        
        else:
            if move == 1:
                prev_r, prev_c, curr_r, curr_c = move_down(prev_r, prev_c, curr_r, curr_c, randomize=randomize)
                move = 0
            else:
                prev_r, prev_c, curr_r, curr_c = move_right(prev_r, prev_c, curr_r, curr_c, randomize=randomize)
                move = 1

        if move == 1:
            for col in range(prev_c, curr_c, int(np.sign(curr_c-prev_c))):
                path_info.append({
                    'row_idx': curr_r,
                    'col_idx': col,
                    'row_name': df.index[curr_r],
                    'col_name': df.columns[col],
                    'value': df.iloc[curr_r, col],
                    'piece': n
                })

        elif move == 0:
            for row in range(prev_r, curr_r, int(np.sign(curr_r-prev_r))):
                path_info.append({
                    'row_idx': row,
                    'col_idx': curr_c,
                    'row_name': df.index[row],
                    'col_name': df.columns[curr_c],
                    'value': df.iloc[row, curr_c],
                    'piece': n
                })

    path_info.append({
        'row_idx': curr_r,
        'col_idx': curr_c,
        'row_name': df.index[curr_r],
        'col_name': df.columns[curr_c],
        'value': df.iloc[curr_r, curr_c],
        'piece': n+1
    })

    return path_info


def color_path(df, path):
    bg = pd.DataFrame('', index=df.index, columns=df.columns)
    #print (path)
    for entry in path:
        bg.loc[entry['row_name'], entry['col_name']] = 'background-color:yellow'
    return bg


# Generate 2D tables with meaningful rows and columns
def generate_business_sales_data(num_rows, num_cols, p=0.5):
    product_names = products
    rows = np.random.choice(product_names, num_rows, replace=False)
    columns = np.random.choice(months, num_cols, replace=False)
    data = np.random.choice(10, size=(num_rows, num_cols)) 
    table = pd.DataFrame(data, index=rows, columns=columns)
    path = zigzag_path(table, p)
    return table.style.apply(lambda t: color_path(t, path), axis=None), path

# Generate multiple tables for each category with varying number of rows and columns
def generate_tables(generator_function, num_tables, min_rows=5, min_cols=5, max_rows=10, max_cols=10, p=0.5):
    tables = []
    paths = []
    for _ in tqdm(range(num_tables)):
        num_rows = np.random.choice(range(min_rows, max_rows+1), 1, replace=True)[0]
        num_cols = np.random.choice(range(min_cols, max_cols+1), 1, replace=True)[0]
        table, path = generator_function(num_rows, num_cols, p)
        tables.append(table)
        paths.append(path)
    return tables, paths


# Function to apply asterisks to specific cells
def mark_cells_with_asterisk(styler, path_info):
    def apply_mark(df):
        # Modify cell values based on path_info
        # We use df.iat to directly access and modify the values
        for cell in path_info:
            i, j = cell['row_idx'], cell['col_idx']
            df.iat[i, j] = f"{df.iat[i, j]}*"  # Append * to the cell value
        return df

    # Apply the modification to the entire DataFrame
    styler = styler.data.copy()  # Work with a copy of the DataFrame to avoid modifying the original
    modified_df = apply_mark(styler)

    # Return the modified Styler object
    return pd.io.formats.style.Styler(modified_df)

# Updated generate_latex_table function to incorporate path_info for marking cells
def generate_latex_table(styler, caption, label, path_info):
    # Apply the style to mark specific cells
    styled = mark_cells_with_asterisk(styler, path_info)
    
    # Generate LaTeX table code from the styled DataFrame
    latex = styled.to_latex()  # escape=False allows special characters like *
    latex = f"\\begin{{table}}[ht]\n\\centering\n{latex}\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}"
    return latex


# Generate LaTeX codes for multiple tables
def generate_latex_codes_for_tables(tables, path_infos, base_caption, base_label):
    latex_codes = []
    for _, (table, path_info) in enumerate(zip(tables, path_infos)):
        caption = f"{base_caption}"
        label = f"{base_label}"
        latex_code = generate_latex_table(table, caption, label, path_info)
        latex_codes.append(latex_code)
    return latex_codes



def get_output(path_info):

    output = []
    digit_to_name = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9'
    }


    all_values = []
    for cell in path_info:
        row_idx, col_idx, row_name, col_name, cell_value, path_segment = cell.values()
        output += [{"row index": str(row_idx+1), "col index": str(1+col_idx), "row name": row_name, "col name": col_name, "value": digit_to_name[cell_value]}]
        all_values += [cell_value]
    total_ = sum([int(v) for v in all_values])
    
    return output, str(total_)


def create_and_save_table(df, path, filename):
    df = df.data
    fig, ax = plt.subplots(figsize=(15, 8))  # Adjust the size as needed
    ax.axis('tight')
    ax.axis('off')
    
    def split_column_names(df):
        new_columns = []
        for col in df.columns:
            try:
                if '-' in col:
                    new_columns.append(col.replace('-', '-\n'))
                else:
                    new_columns.append(col)
            except:
                new_columns.append(col)
        df.columns = new_columns
        
        return df

    df = split_column_names(df)
    

    # Dictionary to map digits to their first three letters of their English names
    digit_to_name = {
        0: 'ZERO',
        1: 'ONE',
        2: 'TWO',
        3: 'THREE',
        4: 'FOUR',
        5: 'FIVE',
        6: 'SIX',
        7: 'SEVEN',
        8: 'EIGHT',
        9: 'NINE'
    }

    # Convert the digits in the dataframe
    df_converted = df.applymap(lambda x: digit_to_name[x])
    # Create table
    table = ax.table(cellText=df_converted.values, colLabels=df_converted.columns, rowLabels=df_converted.index, cellLoc='center', loc='center')
    


    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    
    nrows, ncols = df.shape
    width, height = 1.0 / (ncols + 1), 1.0 / (nrows + 1)

    # Adjust column and row sizes
    path_cells = []
    for entry in path:
        path_cells.append((entry['row_idx'], entry['col_idx']))

    for key, cell in table.get_celld().items():

        cell.set_height(0.1)
        cell.set_width(width)
        
        if key[0] == 0:  # Header cells
            cell.set_facecolor('#40466e')
            cell.set_text_props(weight='bold', color='w')
            
            if '-' not in  cell.get_text().get_text() and len(cell.get_text().get_text()) > 6:
                cell.set_fontsize(14)
        elif key[1] == -1:  # Index cells
            cell.set_facecolor('#40466e')
            cell.set_text_props(weight='bold', color='w')
        elif (key[0]-1, key[1]) in path_cells:  # Path cells   # change: row index starts from 1
            cell.set_facecolor('yellow')
        else:
            cell.set_facecolor('#f5f5f5')
    
    # Ensure table is centered
    #table.scale(1, 15./8.)
    # Adjust layout to remove margins
    # Adjust the layout to avoid clipping
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.tight_layout()
       
    # Convert figure to RGB array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((height, width, 3))

    pil_image = Image.fromarray(image)

    plt.close()
    return image


def convert_to_instruction(all_caption_data):
    all_instructions = []
    for instruction_num, data in enumerate(all_caption_data):
        data["id"] = str(instruction_num) + '_' + str(initial_seed)
        all_instructions += [data]
    return all_instructions


if __name__ == '__main__':
    # Generate multiple tables for each category

    domains = ['business_sales']
    domains = {domain: i for i, domain in enumerate(domains)}
    generator_functions = [generate_business_sales_data,]
    latex_labels = [
        ("Business Sales Data for products in different months", "tab:business_sales"),
    ]
    all_tables = []
    all_paths = []

    # Generate LaTeX codes for each set of tables
    for domain in domains:
        table, path = generate_tables(generator_functions[domains[domain]], \
                                      args.num_data, \
                                      min_rows=args.min_rows,\
                                      min_cols=args.min_cols, \
                                      max_rows=args.max_rows, \
                                      max_cols=args.max_cols)
        all_tables.append(table)
        all_paths.append(path)


    image_counter = 0
    # OCR Questions
    ocr_ = []
    all_image_RGBs = []
    split_counter = 0
    filename = os.path.abspath(os.path.join(args.output_dir, 'Piecewisepath_TestImage_yellow_{}_split{}_new.npy'.format(initial_seed, split_counter)))

    for domain in domains:
        caption, label = latex_labels[domains[domain]]
        for tab, path_info in tqdm(zip(all_tables[domains[domain]], all_paths[domains[domain]])):
            
            data_ = {}
            
            image_arr = create_and_save_table(tab, path_info, filename)
            all_image_RGBs += [image_arr]
            data_['image_RGB'] = [filename, image_counter]
            image_counter += 1
            if image_counter % 100 == 0:
                image_array = np.array(all_image_RGBs)  # Converts the list to a single 4D numpy array
                np.save(filename, image_array)

                split_counter += 1
                filename = os.path.abspath(os.path.join(args.output_dir, 'Piecewisepath_TestImage_yellow_{}_split{}_new.npy'.format(initial_seed, split_counter)))
                image_counter = 0
                all_image_RGBs = []
            
            latex_table = generate_latex_table(tab, caption, label, path_info)
            latex_table = latex_table.replace("*", " *").replace("\\\\\n", "\\\\\n ")
            data_['text'] = latex_table

            reasoning, answer = get_output(path_info)
            data_["reasoning_steps"] = reasoning
            data_["answer"] = answer

            data_["start_product"] = reasoning[0]["row name"]
            data_["start_month"] = reasoning[0]["col name"]
            data_["end_product"] = reasoning[-1]["row name"]
            data_["end_month"] = reasoning[-1]["col name"]
            
            ocr_ += [data_]

    # Assuming 'images' is a list of (height, width, 3) numpy arrays
    if len(all_image_RGBs) > 0:
        image_array = np.array(all_image_RGBs)  # Converts the list to a single 4D numpy array
        np.save(filename, image_array)


    all_instructions = convert_to_instruction(ocr_)
    with open(os.path.join(args.output_dir, 'raw_{}.json'.format(initial_seed)), 'w') as json_file:
        json.dump(all_instructions, json_file, indent=4)  # indent=4 is used to pretty-print the JSON

    
   
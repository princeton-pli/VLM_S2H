import os
import sys
import pandas as pd
import numpy as np
import random
from names import products, months
import random
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning

# from PIL import Image
import warnings

import argparse
from tqdm import tqdm
import json

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--num_data", default=1, type=int)
parser.add_argument("--min_rows", default=5, type=int)
parser.add_argument("--min_cols", default=5, type=int)
parser.add_argument("--max_rows", default=10, type=int)
parser.add_argument("--max_cols", default=10, type=int)
parser.add_argument("--min_pathlen", default=10, type=int)
parser.add_argument("--max_pathlen", default=10, type=int)
parser.add_argument("--output_dir", default=None, type=str)
args = parser.parse_args()

args.output_dir = args.output_dir + "/Minlen{}_Maxlen{}".format(args.min_pathlen, args.max_pathlen)
# image_dir = args.output_dir  + '/images'


initial_seed = args.seed
seed = initial_seed

random.seed(seed)
np.random.seed(seed)

os.makedirs(args.output_dir, exist_ok=True)
# os.makedirs(image_dir, exist_ok=True)


def zigzag_path(df, p=0.5, min_path_len=4):

    path_length = np.random.choice(range(5, 10), 1)[0]
    option = np.random.choice(2, 1)[0]

    start_r = np.random.choice(len(df), 1, replace=False)[0]
    start_c = np.random.choice(len(df.columns), 1, replace=False)[0]

    curr_r, curr_c = start_r, start_c
    path_info = []

    path_info.append({
        'row_idx': curr_r,
        'col_idx': curr_c,
        'row_name': df.index[curr_r],
        'col_name': df.columns[curr_c],
        'value': df.iloc[curr_r, curr_c],
        'piece': -1
    })

    for _ in range(path_length):
        curr_c += 1
        if curr_c >= len(df.columns):
            curr_c = 0
            curr_r += 1

        if curr_r >= len(df):
            break
        
        path_info.append({
            'row_idx': curr_r,
            'col_idx': curr_c,
            'row_name': df.index[curr_r],
            'col_name': df.columns[curr_c],
            'value': df.iloc[curr_r, curr_c],
            'piece': -1
        })

    if option == 1:
        path_info = path_info[::-1]

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
        num_rows = np.random.choice(range(min_rows, max_rows+1), 1)[0]
        num_cols = np.random.choice(range(min_cols, max_cols+1), 1)[0]
        table, path = generator_function(num_rows, num_cols, p)
        tables.append(table)
        paths.append(path)
    return tables, paths


# Function to apply asterisks to specific cells
def mark_cells_with_asterisk(styler, path_info):
    def apply_mark(df):
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


def get_piecewise_output(path_info):
    n_pieces = path_info[-1]["piece"]
    common_output = 'The path is composed of ' + str(n_pieces) + ' linear segments. We enumerate the relevant row indices, column indices, row names, column names, and their corresponding values in each segment separately.\n\n'
    
    all_values = []
    
    start_index = 0
    curr_piece = 0
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


    for n in range(n_pieces):
        seg_values = []
        common_output += 'Segment ' + str(n+1) + ':\n\nRow Index \t Col Index \t Row Name \t Col Name \t Cell Value\n'
        for idx in range(start_index, len(path_info)):
            cell = path_info[idx]
            #cell in path_info:
            row_idx, col_idx, row_name, col_name, cell_value, path_segment = cell.values()
            common_output += str(row_idx+1) + '\t' + str(col_idx+1) + '\t' + str(row_name) + '\t' + str(col_name) + '\t' + digit_to_name[cell_value] + ' (' + str(cell_value) + ')\n'
            seg_values += [cell_value]
            #print (path_segment, curr_piece)
            if path_segment != curr_piece:
                break
        start_index = idx
        curr_piece = path_segment
        common_output += '\n\n'
        
        total_ = sum([int(v) for v in seg_values[:-1]])
        all_values += [total_]
        common_output += 'The total value in this segment, excluding the last cell, is given by ' + ' + '.join([str(v) for v in seg_values[:-2]]) + ' + ' + str(seg_values[-2]) + ' = ' + str(total_) + ".\n\n"
    
    all_values += [seg_values[-1]]
    total_ = sum([int(v) for v in all_values])
    common_output += 'Hence, the total value after adding the values from each segment and the final cell value is ' + ' + '.join([str(v) for v in all_values[:-1]]) + ' + ' + str(all_values[-1]) + ' = ' + str(total_) + ".\n\nAnswer: " + str(total_)
    return common_output



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

    plt.close()
    return image


def convert_to_instruction(all_caption_data):
    all_instructions = []
    for instruction_num, data in enumerate(all_caption_data):
        data["id"] = str(instruction_num) + '_' + str(initial_seed)
        all_instructions += [data]
    return all_instructions


if __name__ == '__main__':

    domains = ['business_sales']
    domains = {domain: i for i, domain in enumerate(domains)}
    generator_functions = [generate_business_sales_data] 
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


    # image_counter = 0
    # OCR Questions
    ocr_ = []
    # all_image_RGBs = []
    # split_counter = 0
    # filename = os.path.abspath(os.path.join(image_dir, 'Linearpath_TestImage_yellow_{}_split{}_new.npy'.format(initial_seed, split_counter)))

    for domain in domains:
        caption, label = latex_labels[domains[domain]]
        for tab, path_info in tqdm(zip(all_tables[domains[domain]], all_paths[domains[domain]])):
            
            data_ = {}
            
            # image_arr = create_and_save_table(tab, path_info, filename)
            # all_image_RGBs += [image_arr]
            # data_['image_RGB'] = [filename, image_counter]
            # image_counter += 1
            # if image_counter % 100 == 0:
                # image_array = np.array(all_image_RGBs)  # Converts the list to a single 4D numpy array
                # np.save(filename, image_array)

                # split_counter += 1
                # filename = os.path.abspath(os.path.join(image_dir, 'Linearpath_TestImage_yellow_{}_split{}_new.npy'.format(initial_seed, split_counter)))
                # image_counter = 0
                # all_image_RGBs = []
            
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
    # if len(all_image_RGBs) > 0:
    #     image_array = np.array(all_image_RGBs)  # Converts the list to a single 4D numpy array
    #     np.save(filename, image_array)


    all_instructions = convert_to_instruction(ocr_)
    with open(os.path.join(args.output_dir, 'raw_{}.json'.format(initial_seed)), 'w') as json_file:
        json.dump(all_instructions, json_file, indent=4)  # indent=4 is used to pretty-print the JSON

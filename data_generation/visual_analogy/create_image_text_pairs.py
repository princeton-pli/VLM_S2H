import os
from utils import *
import numpy as np
import pickle
from matplotlib import pyplot as plt
import json
from tqdm import tqdm
from joblib import Parallel, delayed

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", default='', type=str)
parser.add_argument("--nshot", default=2, type=int)
parser.add_argument("--multi-images", action='store_true')
parser.add_argument("--output-dir", default='', type=str)
parser.add_argument("--num-workers", default=1, type=int)
args = parser.parse_args()

multi_img = 'multi_image' if args.multi_images else 'single_image'
os.makedirs(args.output_dir, exist_ok=True)
match_pattern = {True: 'can suggest', False: 'does not suggest'}

seed = 42


def print_panel(panel_info):
    text = ''
    objects = panel_info.objects
    if len(objects) > 0:
        if type(objects[0]) == Line:
            object_types = ', '.join(set(list(map(lambda x: str(x), panel_info.get_all_types())))) if len(panel_info.get_all_types()) > 0 else 'NA'
            object_colors = ', '.join(set(list(map(lambda x: str(x), panel_info.get_all_colors())))) if len(panel_info.get_all_colors()) > 0 else 'NA'
            text += f"type: {object_types}, color: {object_colors}\n"
        else:  # objects of shape type
            if len(objects) == 1:
                text += f"1 shape; "
            else:
                text += f"{len(objects)} shapes; "
            object_types = ', '.join(set(list(map(lambda x: str(x), panel_info.get_all_types())))) if len(panel_info.get_all_types()) > 0 else 'NA'
            object_colors = ', '.join(set(list(map(lambda x: str(x), panel_info.get_all_colors())))) if len(panel_info.get_all_colors()) > 0 else 'NA'
            object_sizes = ', '.join(set(list(map(lambda x: str(x), panel_info.get_all_sizes())))) if len(panel_info.get_all_sizes()) > 0 else 'NA'
            object_positions = ', '.join(set(list(map(lambda x: str(x), panel_info.get_all_positions())))) if len(panel_info.get_all_positions()) > 0 else 'NA'
            text += f"type: {object_types}, color: {object_colors}, size: {object_sizes}, position: {object_positions}\n"
    return text


def create_text(source_meta_info, query_meta_info, option_meta_info):
    # Create text from the meta information
    text = ""
    for i, example in enumerate(source_meta_info):
        # Each info contains meta information about one example that consists of three panel images
        text += f"Example {i+1}:\n"
        for j, panel_info in enumerate(example):
            text += f"Panel {j+1}:\n"
            text += print_panel(panel_info)
    
    text += "Query:\n"
    for i, panel_info in enumerate(query_meta_info):
        text += f"Panel {i+1}:\n"
        text += print_panel(panel_info)

    text += "Options:\n"
    for i, (_, panel_info) in enumerate(option_meta_info):
        text += f"Option {i+1}:\n"
        text += print_panel(panel_info)
            
    return text


def create_cot(source, query, options, answer, alt_order=False, alt_attributes=False):
    gt_domain, gt_relation = query['label']['domain'], query['label']['relation']
    cot = 'We first analyze the examples by considering the applicable attributes and relations:\n'

    # Identifying patterns in the examples
    example_labels = []
    for i, example in enumerate(source['meta_info']):
        d, r = source['label'][i]['domain'], source['label'][i]['relation']
        example_labels.append(f'({" ".join(d.split("_"))}, {r})')
        object_types= d.split('_')[0]
        attributes = [attr for attr in DOMAINS if attr.startswith(object_types)]
        cot += f'Example {i+1}:\n'
        if alt_order:
            np.random.seed(seed)
            attributes = np.random.permutation(attributes)
        for attr in attributes:
            a = attr.split('_')[1]
            cot += f'{a}:\n'
            for j, panel_info in enumerate(example):
                attr_info = list(map(lambda x: str(x), panel_info.get_all_attributes(a)))
                attr_info = list(set(attr_info))
                if alt_attributes:
                    np.random.seed(seed)
                    attr_info = np.random.permutation(attr_info)
                if len(attr_info) > 0:
                    cot += f'Image {j+1}: {", ".join(attr_info)}\n'
                else:
                    cot += f'Image {j+1}: NA\n'

            if d != attr:
                cot += 'No pattern.\n'
            else:
                cot += f'This suggests the {r} relation.\n'
            
    cot += f'The examples suggest the following patterns: {", ".join(example_labels)}.\n'
    cot += f'We conclude that {gt_relation} is the common relation.\n'

    # Applying the same pattern to the query
    cot += f'Now, we analyze the images in the query:\n'
    object_types= query['label']['domain'].split('_')[0]
    attributes = [attr for attr in DOMAINS if attr.startswith(object_types)]
    for attr in attributes:
        a = attr.split('_')[1]
        cot += f'{a}:\n'
        for i, panel_info in enumerate(query['meta_info']):
            attr_info = list(map(lambda x: str(x), panel_info.get_all_attributes(a)))
            attr_info = list(set(attr_info))  # Remove duplicates
            if len(attr_info) > 0:
                cot += f'Image {i+1}: {", ".join(attr_info)}\n'
            else:
                cot += f'Image {i+1}: NA\n'

    # Analyzing the options
    cot += f'Looking at the options, we find:\n'
    for i, ((d, r), panel_info) in enumerate(options['meta_info']):
        attr_info = set(list(map(lambda x: str(x), panel_info.get_all_attributes(d.split('_')[1]))))
        attr = ' '.join(d.split('_'))
        if len(attr_info) > 0:
            cot += f'Option {i+1} has {attr} of {", ".join(attr_info)}, '
        else:
            cot += f'Option {i+1} does not have {attr}, '
        cot += f'so it is consistent with the {r} relation on {attr}, '
        if i != answer: # incorrect options
            cot += f'but {r} is not the target relation.\n'
        else:   # correct option
            assert(r == gt_relation)
            cot += f'and {r} is the target relation.\n'
    cot += f'We conclude that option {answer+1} is the answer since it is consistent with the {gt_relation} relation.\n'

    return cot


# COT Ablations: create alternative COT templates
def create_cot_alt_template(source, query, options, answer):
    gt_domain, gt_relation = query['label']['domain'], query['label']['relation']
    cot = ''

    for i, example in enumerate(source['meta_info']):
        d, r = source['label'][i]['domain'], source['label'][i]['relation']
        cot += f'Example {i+1}:\n'
        a = ' '.join(d.split('_'))
        cot += f'{a}:\n'
        for j, panel_info in enumerate(example):
            d_info = list(map(lambda x: str(x), panel_info.get_all_attributes(d.split('_')[1])))
            d_info = list(set(d_info))
            if len(d_info) > 0:
                cot += f'Image {j+1}: {", ".join(d_info)}\n'
            else:
                cot += f'Image {j+1}: NA\n'

        cot += f'So example {i+1} follows {r} relation.\n'

    cot += f'Since both examples follow {gt_relation} relation, the query also follows {gt_relation} relation.\n'

    cot += 'Query:\n'
    object_types= query['label']['domain'].split('_')[0]
    attributes = [attr for attr in DOMAINS if attr.startswith(object_types)]
    for attr in attributes:
        a = ' '.join(attr.split('_'))
        cot += f'{a}:\n'
        for i, panel_info in enumerate(query['meta_info']):
            attr_info = list(map(lambda x: str(x), panel_info.get_all_attributes(attr.split('_')[1])))
            attr_info = list(set(attr_info))  # Remove duplicates
            if len(attr_info) > 0:
                cot += f'Image {i+1}: {", ".join(attr_info)}\n'
            else:
                cot += f'Image {i+1}: NA\n'

    for i, ((d, r), panel_info) in enumerate(options['meta_info']):
        attr_info = set(list(map(lambda x: str(x), panel_info.get_all_attributes(d.split('_')[1]))))
        attr = ' '.join(d.split('_'))
        if len(attr_info) > 0:
            cot += f'Option {i+1} has {attr} of {", ".join(attr_info)}, '
        else:
            cot += f'Option {i+1} does not have {attr}, '
        cot += f'so it follows the {r} relation on {attr} with the query panels.\n'
        if i == answer:
            assert(r == gt_relation)
            cot = cot[:-2]
            cot += f'and this is the correct option.\n'

    return cot


def create_answer(source, options, answer):
    response = ""
    # Add answer about the examples
    for i, label in enumerate(source["label"]):
        domain = ' '.join(label["domain"].split('_'))
        response += f'Example {i+1}: ({domain}, {label["relation"]})\n'
    # Add answer about the options
    for i, label in enumerate(options["label"]):
        consistent = "consistent" if i == answer else "not consistent"
        domain = ' '.join(label["domain"].split('_'))
        response += f'Option {i+1}: ({domain}, {label["relation"]}), {consistent}\n'
    # Finally append the answer
    response += f"\nAnswer: {str(answer+1)}"
    return response


def create_image_RGB(examples, query, options):
    fig = plt.figure(layout='constrained', figsize=(8, 9))
    subfigs = fig.subfigures(5, 1, height_ratios=[1,1,1,0.5,1])

    plots = []
    blank_contexts = []
    
    gs0 = subfigs[0].add_gridspec(1, 5, wspace=0, width_ratios=[0.5, 1, 1, 1, 0.5])
    gs0_subplots = gs0.subplots()
    plots += [gs0_subplots[1:-1]]
    blank_contexts += [gs0_subplots[0], gs0_subplots[-1]]

    gs1 = subfigs[1].add_gridspec(1, 5, wspace=0, width_ratios=[0.5, 1, 1, 1, 0.5])
    gs1_subplots = gs1.subplots()
    plots += [gs1_subplots[1:-1]]
    blank_contexts += [gs1_subplots[0], gs1_subplots[-1]]

    gs2 = subfigs[2].add_gridspec(1, 5, wspace=0, width_ratios=[0.5, 1, 1, 1, 0.5])
    gs2_subplots = gs2.subplots()
    plots += [gs2_subplots[1:-1]]
    blank_contexts += [gs2_subplots[0], gs2_subplots[-1]]

    for i in range(len(examples)):
        for j in range(len(examples[0])):
            plots[i][j].imshow(examples[i][j], cmap='gray', vmin=0, vmax=255, aspect='equal')
            plots[i][j].set_xticks([])
            plots[i][j].set_yticks([])

    plots[-1][0].imshow(query[0], cmap='gray', vmin=0, vmax=255, aspect='equal')
    plots[-1][1].imshow(query[1], cmap='gray', vmin=0, vmax=255, aspect='equal')
    plots[-1][2].text(0.5, 0.4, '?', fontsize=50, ha='center')
    plots[-1][2].set_aspect('equal')
    for j in range(3):
        plots[-1][j].set_xticks([])
        plots[-1][j].set_yticks([])

    for x in blank_contexts:
        x.axis('off')

    blank_row = subfigs[3].subplots()
    blank_row.axis('off')

    gs3 = subfigs[4].add_gridspec(1, 4, wspace=0)
    gs3_subplots = gs3.subplots()
    for i in range(len(options)):
        gs3_subplots[i].imshow(options[i], cmap='gray', vmin=0, vmax=255, aspect='equal')
        gs3_subplots[i].set_xticks([])
        gs3_subplots[i].set_yticks([])

    # Convert figure to RGB array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((height, width, 3))

    plt.close()
    return image


def create_image_RGB_separate(examples, query, options):
    fig = plt.figure(layout='constrained', figsize=(6, 6))
    subfigs = fig.subfigures(3, 1, height_ratios=[1,1,1])

    plots = []
    gs0 = subfigs[0].add_gridspec(1, 3, wspace=0)
    plots += [gs0.subplots()]
    gs1 = subfigs[1].add_gridspec(1, 3, wspace=0)
    plots += [gs1.subplots()]
    gs2 = subfigs[2].add_gridspec(1, 3, wspace=0)
    plots += [gs2.subplots()]
    for i in range(len(examples)):
        for j in range(len(examples[0])):
            plots[i][j].imshow(examples[i][j], cmap='gray', vmin=0, vmax=255, aspect='equal')
            plots[i][j].set_xticks([])
            plots[i][j].set_yticks([])

    plots[-1][0].imshow(query[0], cmap='gray', vmin=0, vmax=255, aspect='equal')
    plots[-1][1].imshow(query[1], cmap='gray', vmin=0, vmax=255, aspect='equal')
    plots[-1][2].text(0.5, 0.4, '?', fontsize=50, ha='center')
    plots[-1][2].set_aspect('equal')
    for j in range(3):
        plots[-1][j].set_xticks([])
        plots[-1][j].set_yticks([])
    plt.close()

    fig2 = plt.figure(layout='constrained', figsize=(8, 2))
    subfigs2 = fig2.subfigures(1, 1)
    gs3 = subfigs2.add_gridspec(1, 4, wspace=0)
    gs3_subplots = gs3.subplots()
    for i in range(len(options)):
        gs3_subplots[i].imshow(options[i], cmap='gray', vmin=0, vmax=255, aspect='equal')
        gs3_subplots[i].set_xticks([])
        gs3_subplots[i].set_yticks([])
    
    # Convert figure to RGB array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((height, width, 3))

    plt.close()
    return image


images = []
split_counter = 0
image_counter = 0
files = [file for file in os.listdir(args.input_dir) if file.endswith('.pkl')]

def process_file(file, ind):
    split_counter = 0
    image_counter = 0
    images = []
    all_data_parts = []
    
    data_dir = os.path.join(args.input_dir, file) 
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    
    filename = os.path.abspath(os.path.join(args.output_dir, f'abstract_data_nshot{args.nshot}_{multi_img}_{ind}_{split_counter}.npy'))
    for d in data:
        data_ = {}

        # double check the answer option is the ground truth
        assert(d['options']['label'][d['answer']]['relation'] == d['query']['label']['relation'])

        source = d['source']

        if args.multi_images:
            image = create_image_RGB_separate(source['image'], d['query']['image'], d['options']['image'])
        else:
            image = create_image_RGB(source['image'], d['query']['image'], d['options']['image'])
        images.append(image)
        data_['image'] = [filename, image_counter]
        image_counter += 1

        data_['text'] = create_text(source['meta_info'], d['query']['meta_info'], d['options']['meta_info'])
        data_['cot_alt_template'] = create_cot_alt_template(source, d['query'], d['options'], d['answer'])
        data_['cot'] = create_cot(source, d['query'], d['options'], d['answer'])
        data_['cot_alt_order'] = create_cot(source, d['query'], d['options'], d['answer'], alt_order=True)
        data_['cot_alt_attributes'] = create_cot(source, d['query'], d['options'], d['answer'], alt_attributes=True)
        data_['answer'] = create_answer(source, d['options'], d['answer'])

        data_['image2ans_question'] = f"The image shows a a puzzle in a 3 by 3 grid followed by 4 options. The puzzle consists of 2 examples (row 1 and 2), a query (row 3), and four options. Each example contains three images following a relation along certain attribute, and this relation is consistent across all examples. The query contains two images. Analyze the changes in the following attributes for each example: line type, line color, shape type, shape color, shape size, shape quantity, shape position, and consider the relations: Progression, XOR, OR, and AND. Progression requires the value of a certain attribute to strictly increase or decrease, but not necessarily by a fixed amount. Please provide your predictions in the format 'Example i: (attribute, relation)' for each example and similarly for options. Provide the final answer as 'Answer: [correct option]'."

        data_['text2ans_question'] = f"The paragraph above describes a puzzle. The puzzle consists of 2 examples, a query, and four options. Each example contains three images following a relation along certain attribute, and this relation is consistent across all examples. The query contains two images. Analyze the changes in the following attributes for each example: line type, line color, shape type, shape color, shape size, shape quantity, shape position, and consider the relations: Progression, XOR, OR, and AND. Progression requires the value of a certain attribute to strictly increase or decrease, but not necessarily by a fixed amount. Please provide your predictions in the format 'Example i: (attribute, relation)' for each example and similarly for options. Provide the final answer as 'Answer: [correct option]'."

        data_['image2text_question'] = f"The image shows a puzzle in a 3 by 3 grid followed by 4 options. Convert the image into a text version of the puzzle that describes each image in the example (row 1 and 2), query (row 3), and options (at the bottom) by identifying the value of the following attributes, if applicable: line type, line color, shape type, shape color, shape size, shape quantity, and shape position."

        data_['seed'] = d['seed']

        all_data_parts += [data_]

        if image_counter % 100 == 0:
            images_array = np.array(images)  # Converts the list to a single 4D numpy array
            np.save(filename, images_array)
            print(f"Saved {filename}")

            images = []
            split_counter += 1
            image_counter = 0
            filename = os.path.abspath(os.path.join(args.output_dir, f'abstract_data_nshot{args.nshot}_{multi_img}_{ind}_{split_counter}.npy'))

    if len(image) > 0:
        images_array = np.array(images)  # Converts the list to a single 4D numpy array
        np.save(filename, images_array)
        print(f"Saved {filename}")

    return all_data_parts

# Run parallel job for processing file for file in files
indexed_files = []
for i, file in enumerate(files):
    indexed_files.append((file, i))

for file, index in indexed_files:
    print(f"Processing file {file} with index {index}")

all_data_gathered = Parallel(n_jobs=args.num_workers, verbose=50)(delayed(process_file)(file, index) for [file, index] in indexed_files)

all_data = []
for data in all_data_gathered:
    all_data += data


# Save the instructions to a single JSON file
with open(os.path.join(args.output_dir, f'raw.json'), 'w') as json_file:
    json.dump(all_data, json_file, indent=4)

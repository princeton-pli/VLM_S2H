import json
import random
import argparse
import os
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", default='', type=str)
parser.add_argument("--output-dir", default='', type=str)
parser.add_argument("--eval", action='store_true')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


def load_data(file_path, filename):
    file_path = os.path.join(file_path, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def convert_to_instruction(all_caption_data, prompt_image, prompt_text, prompt_type, text=False, cot=True, image_RGB=True, cot_alt_template=False, cot_alt_order=False, cot_alt_value=False):
    all_instructions = []
    for instruction_num, data in enumerate(all_caption_data):
        instruction = {}
        instruction["id"] = str(instruction_num)
        instruction["image_RGB"] = data['image'] if image_RGB else ""

        conversation = []
        conv = {}
        conv["from"] = "human"
        if prompt_image:
            conv["value"] = "<image>\n\n" 
        if prompt_text:
            conv["value"] = data['text'] + "\n"
        conv["value"] += data[f"{prompt_type}_question"]
        conversation += [conv]

        conv = {}
        conv["from"] = "gpt"
        conv["value"] = ""
        if text:
            conv["value"] += "Convert the image into a text version of the puzzle.\n" + data['text'] + "\n"
        if cot:   
            if cot_alt_template:
                conv["value"] += data['cot_alt_template'] + "\n\n"
            elif cot_alt_order:
                conv["value"] += data['cot_alt_order'] + "\n\n"
            elif cot_alt_value:
                conv["value"] += data['cot_alt_attributes'] + "\n\n"
            else:
                conv["value"] += data['cot'] + "\n\n"
        if prompt_type == "image2text":
            conv["value"] = conv["value"][:-2]
        else:
            conv["value"] += data["answer"]
        conversation += [conv]
        instruction["conversations"] = conversation

        all_instructions += [instruction]

    return all_instructions


def create_data(input_dir, output_dir, eval=False):
    all_caption_data = load_data(input_dir, "raw.json")

    if eval:
        all_caption_data = all_caption_data[:500]

    # image2answer+cot
    image2answer_cot = convert_to_instruction(all_caption_data, prompt_image=True, prompt_text=False, prompt_type="image2ans", cot=True)
    with open(os.path.join(output_dir, 'image2answer+cot.json'), 'w') as f:
        json.dump(image2answer_cot, f, indent=4)

    # image2answer (no cot)
    image2answer = convert_to_instruction(all_caption_data, prompt_image=True, prompt_text=False, prompt_type="image2ans", cot=False)
    with open(os.path.join(output_dir, 'image2answer.json'), 'w') as f:
        json.dump(image2answer, f, indent=4)

    # image2answer+text+cot
    image2answer_text_cot = convert_to_instruction(all_caption_data, prompt_image=True, prompt_text=False, prompt_type="image2ans", text=True, cot=True)
    with open(os.path.join(output_dir, 'image2answer+text+cot.json'), 'w') as f:
        json.dump(image2answer_text_cot, f, indent=4)

    # image2answer+text (no cot)
    image2answer_text = convert_to_instruction(all_caption_data, prompt_image=True, prompt_text=False, prompt_type="image2ans", text=True, cot=False)
    with open(os.path.join(output_dir, 'image2answer+text.json'), 'w') as f:
        json.dump(image2answer_text, f, indent=4)

    # image2text
    image2text = convert_to_instruction(all_caption_data, prompt_image=True, prompt_text=False, prompt_type="image2text", text=True, cot=False)
    with open(os.path.join(output_dir, 'image2text.json'), 'w') as f:
        json.dump(image2text, f, indent=4)
    with open(os.path.join(output_dir, 'image2text+cot.json'), 'w') as f:
        json.dump(image2text, f, indent=4)

    # text2answer+cot
    text2answer_cot = convert_to_instruction(all_caption_data, prompt_image=False, prompt_text=True, prompt_type="text2ans", cot=True, image_RGB=False)
    with open(os.path.join(output_dir, 'text2answer+cot.json'), 'w') as f:
        json.dump(text2answer_cot, f, indent=4)

    # text2answer (no cot)
    text2answer = convert_to_instruction(all_caption_data, prompt_image=False, prompt_text=True, prompt_type="text2ans", cot=False, image_RGB=False)
    with open(os.path.join(output_dir, 'text2answer.json'), 'w') as f:
        json.dump(text2answer, f, indent=4)

    # text2answer+cot (alternate cot template)
    text2answer_cot_template = convert_to_instruction(all_caption_data, prompt_image=False, prompt_text=True, prompt_type="text2ans", cot=True, image_RGB=False, cot_alt_template=True)
    with open(os.path.join(output_dir, 'text2answer+cot-template.json'), 'w') as f:
        json.dump(text2answer_cot_template, f, indent=4)

    # text2answer+cot (alternate cot order)
    text2answer_cot_order = convert_to_instruction(all_caption_data, prompt_image=False, prompt_text=True, prompt_type="text2ans", cot=True, image_RGB=False, cot_alt_order=True)
    with open(os.path.join(output_dir, 'text2answer+cot-order.json'), 'w') as f:
        json.dump(text2answer_cot_order, f, indent=4)

    # text2answer+cot (alternate cot attribute value)
    text2answer_cot_value = convert_to_instruction(all_caption_data, prompt_image=False, prompt_text=True, prompt_type="text2ans", cot=True, image_RGB=False, cot_alt_value=True)
    with open(os.path.join(output_dir, 'text2answer+cot-value.json'), 'w') as f:
        json.dump(text2answer_cot_value, f, indent=4)




create_data(args.input_dir, args.output_dir, eval=args.eval)
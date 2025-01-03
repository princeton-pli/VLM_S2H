import json
import numpy as np
import argparse
import os

data_type_map = {
    "image2text": 0,
    "text2answer": 1,
    "image2answer": 2,
    "image2answer+text": 3
}

def get_data(base_dir, data_type, cot_tag, domain, num_data):
    if num_data <= 0:
        return []
    if data_type == "text2answer" and args.alt_cot_template:
        alt_cot_tag = "-template"
    elif data_type == "text2answer" and args.alt_cot_order:
        alt_cot_tag = "-order"
    elif data_type == "text2answer" and args.alt_cot_value:
        alt_cot_tag = "-value"
    else:
        alt_cot_tag = ""
    filename = data_type + cot_tag + alt_cot_tag + ".json"
    with open(os.path.join(base_dir, domain, filename)) as f:
        data = json.load(f)
    np.random.seed(42)
    np.random.shuffle(data)
    data = data[:num_data]
    return data


def remove_cot_ar(data, cot_ratio):
    new_data = []
    for d in data:
        cot, answer = d["conversations"][1]["value"].split("\n\n\n")
        convert = None
        if cot.startswith("Convert"):
            convert, cot = cot.split("\n\n")
        cot = cot[:int(len(cot) * cot_ratio)]
        if len(cot) > 0:
            d["conversations"][1]["value"] = f"{cot}\n\n\n{answer}"
        else:
            d["conversations"][1]["value"] = f"{answer}"
        if convert:
            d["conversations"][1]["value"] = f"{convert}\n\n{d['conversations'][1]['value']}"
        new_data.append(d)
    return new_data

def add_think(data, num=1):
    new_data = []
    for d in data:
        answer = d["conversations"][1]["value"]
        all_convert = ""
        for i in range(num):
            all_convert += "<Convert> "
        d["conversations"][1]["value"] = all_convert + answer
        new_data.append(d)
    return new_data


def internalize_cot_ar(data):
    ar_index = []
    for i, d in enumerate(data):
        if "id" in d:
            ar_index.append(i)
    data = np.array(data)
    new_data = data[ar_index]
    num_cot = 0.3 * len(new_data)
    num_no_cot = 0.3 * len(new_data)
    num_internalize = len(new_data) - num_cot - num_no_cot

    cot_data = new_data[:int(num_cot)]
    internalize_data = new_data[int(num_cot):int(num_cot + num_internalize)]
    no_cot_data = new_data[int(num_cot + num_internalize):]
    new_internalize_data = []
    num_level = 100
    split = int(len(internalize_data) / (num_level+1))
    for i in range(num_level+1):
        if i == num_level:
            new_internalize_data += remove_cot_ar(internalize_data[i*split:], (num_level - i) / 10)
            print(new_internalize_data[-1]["conversations"][1]["value"])
        else:
            new_internalize_data += remove_cot_ar(internalize_data[i*split:(i+1)*split], (num_level - i) / 10)
    no_cot_data = remove_cot_ar(no_cot_data, 0)

    new_data = cot_data.tolist() + new_internalize_data + no_cot_data
    data[ar_index] = new_data
    return data.tolist()


## returns dictionary of arrays of whether the following data type is included
## { "SIMPLE": [I->T, T->S, I->S, I->T+S], "HARD": [I->T, T->S, I->S, I->T+S] }
def get_data_composition(option):

    # Train only on SIMPLE examples
    ## EPOCHS = 3
    if option == 1: # Image-via-text supervision
        return {"SIMPLE": [False, False, False,  True], "HARD": [False, False, False, False]}
    
    ## EPOCHS = 2
    if option == 2: # Text supervision
        return {"SIMPLE": [False,  True, False, False], "HARD": [False, False, False, False]}
    
    ## EPOCHS = 2
    if option == 3: # Image supervision with epochs 2
        return {"SIMPLE": [False, False,  True, False], "HARD": [False, False, False, False]}

    ## EPOCHS = 3
    if option == 4: # Image supervision with epochs 3
        return {"SIMPLE": [False, False,  True, False], "HARD": [False, False, False, False]}

    ## EPOCHS = 1.5
    if option == 5: # Image+Text supervision with epochs 1.5
        return {"SIMPLE": [False,  True,  True, False], "HARD": [False, False, False, False]}

    ## EPOCHS = 1
    if option == 6: # Mix supervision with epochs 1
        return {"SIMPLE": [False,  True,  True,  True], "HARD": [False, False, False, False]}

    # Train on SIMPLE+HARD examples
    if option == 7: # Mix+ supervision with epochs 1
        return {"SIMPLE": [False,  True,  True,  True], "HARD": [False,  True, False, False]}
 
    if option == 8: # First phase of supervision for Align-Mix+ 
        return {"SIMPLE": [False,  True, False,  True], "HARD": [False, False, False, False]}

    if option == 9: # Image-via-Text+; 3 epochs on SIMPLE image
        return {"SIMPLE": [False, False, False,  True], "HARD": [False,  True, False, False]}

def main(args):
    base_dir = os.path.join("synthetic_data/", args.setting)
    if args.alt_cot_template or args.alt_cot_order or args.alt_cot_value:
        base_dir += "_cot_ablations"
    if args.no_cot:
        cot_tag = ""
    else:
        cot_tag = "+cot"

    if args.alt_cot_template:
        alt_cot_tag = "-template"
    elif args.alt_cot_order:
        alt_cot_tag = "-order"
    elif args.alt_cot_value:
        alt_cot_tag = "-value"
    else:
        alt_cot_tag = ""

    ### LATER ADD FOR MULTI EPOCH DATASETS
    if args.option in [2,3]:
        args.total_num_data = args.total_num_data // 2

    if args.option in [1,4]:
        args.total_num_data = args.total_num_data // 3
    
    if args.option in [5]:
        args.total_num_data = 2 * args.total_num_data // 3

    
    data_composition = get_data_composition(args.option)
    total_num_data = args.total_num_data * 1000
    num_data_dict = {
        "SIMPLE": total_num_data // (any(data_composition["SIMPLE"]) + any(data_composition["HARD"])) * any(data_composition["SIMPLE"]),
        "HARD": total_num_data // (any(data_composition["SIMPLE"]) + any(data_composition["HARD"])) * any(data_composition["HARD"]),
    }
    
    result = []
    out_filename = ""
    
    
    if args.option in [9]:
        for domain in ["SIMPLE"]:
            num_data = num_data_dict[domain]
            print("------------------------------")
            print("{}: {}".format(domain, num_data))
            if num_data <= 0: continue
            for data_type, idx in data_type_map.items():
                num_data_ = num_data // sum(data_composition[domain]) * data_composition[domain][idx]
                if num_data_ <= 0: continue
                print("{}: {}".format(data_type, num_data_ // 3))
                if len(out_filename) > 0:
                    out_filename += "_"
                out_filename += "{}{}+{}-{}k".format(data_type, cot_tag, domain, num_data_ // 3000)
                if not args.get_data_path_only:
                    result += get_data(base_dir, data_type, cot_tag, domain, num_data_ // 3)
        
        result += (result + result)
        for domain in ["HARD"]:
            num_data = num_data_dict[domain]
            print("------------------------------")
            print("{}: {}".format(domain, num_data))
            if num_data <= 0: continue
            for data_type, idx in data_type_map.items():
                num_data_ = num_data // sum(data_composition[domain]) * data_composition[domain][idx]
                if num_data_ <= 0: continue
                print("{}: {}".format(data_type, num_data_))
                if len(out_filename) > 0:
                    out_filename += "_"
                out_filename += "{}{}+{}-{}k".format(data_type, cot_tag, domain, num_data_ // 1000)
                if not args.get_data_path_only:
                    result += get_data(base_dir, data_type, cot_tag, domain, num_data_)
    else:
        for domain in ["SIMPLE", "HARD"]:
            num_data = num_data_dict[domain]
            print("------------------------------")
            print("{}: {}".format(domain, num_data))
            if num_data <= 0: continue
            for data_type, idx in data_type_map.items():
                num_data_ = num_data // sum(data_composition[domain]) * data_composition[domain][idx]
                if num_data_ <= 0: continue
                print("{}: {}".format(data_type, num_data_))
                if len(out_filename) > 0:
                    out_filename += "_"
                if data_type == "text2answer":
                    out_filename += "{}{}{}+{}-{}k".format(data_type, cot_tag, alt_cot_tag, domain, num_data_ // 1000)
                else:
                    out_filename += "{}{}+{}-{}k".format(data_type, cot_tag, domain, num_data_ // 1000)
                if not args.get_data_path_only:
                    result += get_data(base_dir, data_type, cot_tag, domain, num_data_)
    
    np.random.shuffle(result)
    if args.internalize_cot_ar:
        result = internalize_cot_ar(result)
        out_filename += "_interncot"
    out_filename += ".json"
    if not args.get_data_path_only:
        print (len(result))
        os.makedirs(os.path.join(base_dir, "json_files"), exist_ok=True)
        with open(os.path.join(base_dir, "json_files", out_filename), "w") as f:
            json.dump(result, f, indent=2)
            print("Saved data to:", os.path.join(base_dir, "json_files", out_filename))
    print("------------------------------")
    print(os.path.join(base_dir, "json_files", out_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, required=True) ## "table_readout" or "grid_navigation"
    parser.add_argument('--no_cot', action="store_true", default=False)
    parser.add_argument('--sequential', action="store_true", default=False)
    parser.add_argument('--option', type=int, required=True)  ## e.g., 1 -> Strategy 1
    parser.add_argument('--total_num_data', type=int, required=True)  ## e.g., 60, 120, 240
    parser.add_argument('--get_data_path_only', action="store_true", default=False) ## If you want to only get a sample data_path
    parser.add_argument('--internalize_cot_ar', action="store_true", default=False) ## If you want to internalize cot_ar
    parser.add_argument('--alt_cot_template', action="store_true", default=False) ## If you want to use alternative cot template
    parser.add_argument('--alt_cot_order', action="store_true", default=False) ## If you want to use alternative cot order
    parser.add_argument('--alt_cot_value', action="store_true", default=False) ## If you want to use alternative cot value
    args = parser.parse_args()
    np.random.seed(42)
    main(args)
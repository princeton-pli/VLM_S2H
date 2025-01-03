import os
from utils import *
import numpy as np
import pickle
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--simple-data", default=120000, type=int, help="number of SIMPLE data samples to generate")
parser.add_argument("--hard-data", default=120000, type=int, help="number of HARD data samples to generate")
parser.add_argument("--simple-eval-data", default=100, type=int, help="number of SIMPLE evaluation data samples to generate")
parser.add_argument("--hard-eval-data", default=100, type=int, help="number of HARD evaluation data samples to generate")

parser.add_argument("--simple", action='store_true', help="specify if SIMPLE data should be generated")
parser.add_argument("--hard", action='store_true', help="specify if HARD data should be generated")

parser.add_argument("--nshot", default=2, type=int, help="number of samples to generate")
parser.add_argument("--output_dir", default="", type=str, help="directory to save the generated data")
parser.add_argument("--seed", default=0, type=int, help="seed for random number generator")
args = parser.parse_args()


SEED = args.seed
np.random.seed(SEED)


hard_combi = [('shape_size', 'XOR'), ('shape_position', 'OR'), ('line_color', 'AND'), ('shape_type', 'Progression'), ('shape_position', 'XOR'), ('shape_type', 'OR'), ('shape_color', 'AND'), ('line_color', 'Progression')]     # held out combinations

all_combi = [(d, r) for d in DOMAINS for r in RELATIONS]
excludes = [('shape_quantity', 'AND'), ('shape_quantity', 'OR'), ('shape_quantity', 'XOR')]     # logical relations other than progression are not defined on shape quantity
all_combi = set(all_combi).difference(excludes)
simple_combi = list(all_combi.difference(hard_combi))
print(f"{len(all_combi)} total combinations")
print(f"{len(hard_combi)} held out combinations: {hard_combi}")
print(f"{len(simple_combi)} in-distribution combinations: {simple_combi}")

simple_output_dir = os.path.join(args.output_dir, 'SIMPLE', 'raw_files')
os.makedirs(simple_output_dir, exist_ok=True)
simple_eval_output_dir = os.path.join(args.output_dir, 'SIMPLE_eval', 'raw_files')
os.makedirs(simple_eval_output_dir, exist_ok=True)
hard_output_dir = os.path.join(args.output_dir, 'HARD', 'raw_files')
os.makedirs(hard_output_dir, exist_ok=True)
hard_eval_output_dir = os.path.join(args.output_dir, 'HARD_eval', 'raw_files')
os.makedirs(hard_eval_output_dir, exist_ok=True)


def get_data(combi, num_data, output_dir, is_HARD=False, seed=SEED):
    domain_mixed = is_HARD
    count = 0
    filecount = 0
    all_data = []
    for count in tqdm(range(num_data)):
        np.random.seed(seed)
        query_d, query_r = combi[np.random.choice(len(combi))]
        n_seed_attempts = 0
        n_seed_success = False

        if domain_mixed:   # choose different domains for each of nshot examples if possible
            example_domains = [(dd, rr) for dd, rr in combi if rr == query_r] 
            if len(example_domains) < args.nshot:       # not enough to sample distinct domain for each
                example_domains = np.array(example_domains)[np.random.choice(len(example_domains), args.nshot, replace=True)]
            else:
                example_domains = np.array(example_domains)[np.random.choice(len(example_domains), args.nshot, replace=False)]
        else:   # choose the same domain for all nshot examples
            example_domains = [(query_d, query_r)] * args.nshot

        while (not n_seed_success):
            n_seed_success = False
            n_seed_attempts = 0
            while (not n_seed_success) and n_seed_attempts < 20:
                # Generate examples
                source_panels = {'meta_info': [], 'image': [], 'label': []}
                progression_direction = None
                nshot_success = False
                
                for i in range(args.nshot):
                    n_samples = 1      
                    example_success = False
                    d, r = example_domains[i]

                    # Generate using rejection sampling
                    while (not example_success) and n_samples < 20:
                        try:
                            panels, options, gt = create_panels(d, r)
                            if len(options) < 1:
                                n_samples += 1
                                continue
                        except:
                            n_samples += 1
                            continue

                        assert(options[0][0] == (d, r))    # double check the first option is the ground truth
                        example_success = True
                        progression_direction = panels[0].get_progression_direction() if (not progression_direction) and r == 'progression' else None
                        panels.append(options[gt][1])
                        source_panels["meta_info"].append(panels)
                        source_panels["image"].append([panel.draw() for panel in panels])
                        source_panels["label"].append({'domain': d, 'relation': r})
                nshot_success = (len(source_panels["meta_info"]) == args.nshot)

                # Generate query only after examples are successfully generated
                query_success = False
                if nshot_success:
                    query_panels = {'meta_info': [], 'image': [], 'label': []}
                    option_panels = {'meta_info': [], 'image': [], 'label': []}
                    n_samples = 1
                    query_success = False
                    while (not query_success) and n_samples < 20:
                        try:
                            if not is_HARD and len(hard_combi) > 0:
                                target_panels, target_options, target_gt = create_panels(query_d, query_r, create_option=True, progression_direction=progression_direction, option_patterns=combi)
                            else:
                                target_panels, target_options, target_gt = create_panels(query_d, query_r, create_option=True, progression_direction=progression_direction)
                            if len(target_options) < 4:    # need 4 options
                                n_samples += 1
                                continue
                        except:
                            continue

                        assert(target_options[target_gt][0] == (query_d, query_r))    # double check the gt option is the ground truth
                        query_success = True
                        query_panels["meta_info"] = target_panels
                        query_panels["image"] = [panel.draw() for panel in target_panels]
                        query_panels["label"] = {'domain': query_d, 'relation': query_r}
                        option_panels["meta_info"] = target_options
                        option_panels["image"] = [option[1].draw() for option in target_options]
                        option_panels["label"] = [{'domain': option[0][0], 'relation': option[0][1]} for option in target_options]

                # Save the generated data if both examples and query are successfully generated
                if nshot_success and query_success:
                    n_seed_success = True
                    data = {
                        'source': source_panels,
                        'query': query_panels,
                        'options': option_panels,
                        'answer': target_gt,
                        'seed': seed
                    }
                    all_data.append(data)
                    count += 1

                    if count % 1000 == 0:
                        filename = f"data_{filecount}.pkl"
                        with open(os.path.join(output_dir, filename), 'wb') as f:
                            pickle.dump(all_data, f) 
                        print(f"Saved data sample {count}/{num_data} {output_dir}, seed={seed}")  
                        filecount += 1
                        all_data = []
                n_seed_attempts += 1
        
            seed += 1
    
    if nshot_success and query_success and len(all_data) > 0:
        filename = f"data_{filecount}.pkl"
        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(all_data, f) 
        print(f"Saved data sample {count}/{num_data} {output_dir}, seed={seed}") 
    
    return seed

# Generate SIMPLE data
if args.simple:
    simple_seed = get_data(simple_combi, args.simple_data, simple_output_dir, is_HARD=False)
    get_data(simple_combi, args.simple_eval_data, simple_eval_output_dir, seed=simple_seed+1, is_HARD=False)
    
# Generate HARD data
if args.hard:
    hard_seed = get_data(hard_combi, args.hard_data, hard_output_dir, is_HARD=True)
    get_data(hard_combi, args.hard_eval_data, hard_eval_output_dir, seed=hard_seed+1, is_HARD=True)

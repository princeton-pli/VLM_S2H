Our code in this directory is adapted from the original repo [VLMEvalKit](https://github.com/open-compass/VLMEvalKit/tree/8106439c2d3b07353c84374bebdd947d4ec16a8f)

## Chages Made

`vlmeval/api/claude.py` (line 7, lines 15-20, line 45, line 47, line 100)

`vlmeval/config.py` (line 58)

- use anthropic API directly
- add support for Claude 3.5 Sonnet

`vlmeval/smp/file.py` (line 17-18)

- save images in project folder instead of user home folder

`vlmeval/smp/vlm.py` (lines 59-61)

- save images from numpy array data 

`vlmeval/utils/dataset.py` (lines 147-149)

`vlmeval/vlm/base.py` (lines 46-47)

- add support to text only evaluation

`vlmeval/evaluate/__init__.py` (line 10)

`vlmeval/evaluate/OurEval.py` (newly added)

`vlmeval/build_our_data.py` (newly added)

`vlmeval/utils/dataset_config.py` (lines 136-137)

`run.py` (lines 151-156)

- add support to evaluation on our synthetic datasets

`vlmeval/vlm/eagle` (newly added)

`vlmeval/vlm/__init__.py` (line 34)

`vlmeval/config.py` (lines 146-150, line 161)

- add support to Eagle
- add support for llama3 (adapted from Cambrian codebase)
- also see changes made to Eagle github repo


## Chages Made for Qwen2.5-VL (adapted from a future version of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit/tree/2d98019d801b92b2a0f02cce1b019a792e296a94/vlmeval/vlm))

`vlmeval/smp/misc.py` (lines 193 - 217)

`vlm/qwen2_vl`

`vlmeval/vlm/__init__.py` (line 20)

`vlmeval/config.py` (lines 150-153)
Our code in this directory is adapted from the original repo [EAGLE](https://github.com/NVlabs/EAGLE/tree/51da2833a98f7ce185d11f945ce035b97cf80314)

## Changes Made
`eagle/model/multimodal_encoder/multi_backbone_channel_concatenation_encoder.py` (lines 56, 97)

`eagle/model/multimodal_encoder/pix2struct_encoder.py` (lines 71, 74)

`eagle/model/multimodal_encoder/sam_encoder.py` (lines 95, 97)
- add a local path to the downloaded model (instead of the online huggingface repo)
- necessary only if you plan on using more visual encoders

`eagle/model/eagle_arch.py` (lines 83, 89)
- remove unnecessary, buggy code on FSDP

`eagle/model/builder.py` (lines 112-116)
- debug buggy code on dtype

`eagle/train/eagle_trainer.py` (lines 149-151)

`train.py` (line 131)
- add support for SequentialSampler

`eagle/mm_utils.py` (lines 222-237)

`eagle/conversation.py` (line 32, lines 111-134, lines 321-333, line 445)

`train.py` (line 53, lines 352-446, lines 744-745, lines 1060-1062, line 1065)
- add support for Llama3 (adapted from Cambrian codebase)

`train.py` (line 57, lines 788-792)
- add support for non-json files

`train.py` (lines 814-819, lines 829-850, lines 860-864, lines 891-892, lines 898-899)
- add support for text-only training

`train.py` (line 1131)
- do not train if final checkpoint already exists

`train_mem.py` (lines 1-7)
- debug potentially buggy code on transformer_engine

## Generalizing from SIMPLE to HARD Visual Reasoning: Can We Mitigate Modality Imbalance in VLMs?

This repository contains the code and pruned models for our paper [Generalizing from SIMPLE to HARD Visual Reasoning: Can We Mitigate Modality Imbalance in VLMs?]()

**************************** **Updates** ****************************
* 01/06/2025: We released [our paper](). Check it out!

## Quick Links

- [Generalizing from SIMPLE to HARD Visual Reasoning: Can We Mitigate Modality Imbalance in VLMs?](#generalizing-from-simple-to-hard-visual-reasoning-can-we-mitigate-modality-imbalance-in-vlms)
- [Quick Links](#quick-links)
- [Overview](#overview)
- [Main Results](#main-results)
- [Experiments](#experiments)
  - [Prepare Conda Environment](#prepare-conda-environment)
  - [Prepare EAGLE Data (Requires 2.5M Data Files + 1TB Storage)](#prepare-eagle-data-requires-25m-data-files--1tb-storage)
  - [Prepare EAGLE-X2-Llama3-8B](#prepare-eagle-x2-llama3-8b)
  - [Prepare Synthetic Data (Highly Recommend Multi-thread Processing)](#prepare-synthetic-data-highly-recommend-multi-thread-processing)
  - [Prepare Evaluation Data](#prepare-evaluation-data)
  - [Train / Evaluate on Synthetic Data](#train--evaluate-on-synthetic-data)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## Overview

## Main Results

## Experiments

In the following section, we provide instructions on reproducing the experiments in our paper.

### Prepare Conda Environment

First set the following bash variable based on your machine and update one of the files.
```Shell
PROJECT_DIR="/absolute path to the project foler/VLM_S2H"
sed -i "s#CHECKPOINTS_ROOT = None#CHECKPOINTS_ROOT = '${PROJECT_DIR}/checkpoints'#" $PROJECT_DIR/VLMEvalKit/vlmeval/config.py
```

Then prepare a conda environment 
```Shell
cd $PROJECT_DIR
conda create -n VLM_S2H python=3.10 -y
conda activate VLM_S2H
pip install pip==24.3.1  # enable PEP 660 support 
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e VLMEvalKit # requirement: pip < 25.0
```

### Prepare EAGLE Data (Requires 2.5M Data Files + 1TB Storage)

First prepare the pretraining data from LLaVA
Note: some of the images in the chat.json file may no longer be available in the images.zip due to copyright.
You may have to delete some of the entries in the json file if the images are not available.
```Shell
cd $PROJECT_DIR/pretraining_data
git lfs install
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K
cd LLaVA-CC3M-Pretrain-595K
unzip images.zip
```

Next prepare the visual instruction tuning data from EAGLE
```Shell
cd $PROJECT_DIR/pretraining_data
git lfs install
git clone https://huggingface.co/datasets/shi-labs/Eagle-1.8M
cd Eagle-1.8M
cat images.tar.part_* > images.tar.gz
tar -xvzf images.tar.gz
```

### Prepare EAGLE-X2-Llama3-8B

fill out cluster-specific details in bash script

```Shell
cd $PROJECT_DIR
sbatch scripts/prepare_eagle/pretrain-eagle-x2-llama3-8b.sh
sbatch scripts/prepare_eagle/finetune-eagle-x2-llama3-8b-1.8m.sh
```

### Prepare Synthetic Data (Highly Recommend Multi-thread Processing)

Each setting may take up to 24 hours without multi-thread processing.
Highly recommend creating a custom bash script and splitting the load.

```Shell
cd $PROJECT_DIR/data_generation/consecutive_table_readout
source generate_data.sh
source generate_eval_data.sh
```

```Shell
cd $PROJECT_DIR/data_generation/table_readout
source generate_data.sh
source generate_eval_data.sh
```

```Shell
cd $PROJECT_DIR/data_generation/grid_navigation
source generate_data.sh
source generate_eval_data.sh
```

```Shell
cd $PROJECT_DIR/data_generation/visual_analogy
source generate_data.sh
```

### Prepare Evaluation Data

This converts the .json file (in the visual instruction tuning data format) to .tsv file (that the VLMEvalKit accepts)

```Shell
cd $PROJECT_DIR/VLMEvalKit
python -m vlmeval.build_our_data
```

### Train / Evaluate on Synthetic Data

fill out cluster-specific details in bash script

```Shell
cd $PROJECT_DIR
source scripts/launcher.sh # see the script for example usage
```

## Bugs or Questions?

## Citation

Please cite our paper if you find this repo helpful:
```bibtex

```

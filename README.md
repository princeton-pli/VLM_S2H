# PREPARE CONDA ENVIRONMENT
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

# PREPARE EAGLE DATA (REQUIRES 2.5M DATA FILES + 1TB STORAGE)
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

# PREPARE EAGLE-X2-LLAMA3-8B
fill out cluster-specific details in bash script

```Shell
cd $PROJECT_DIR
sbatch scripts/prepare_eagle/pretrain-eagle-x2-llama3-8b.sh
sbatch scripts/prepare_eagle/finetune-eagle-x2-llama3-8b-1.8m.sh
```

# PREPARE SYNTHETIC DATA (HIGHLY RECOMMEND MULTI-THREAD PROCESSING)
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

# PREPARE EVALUATION DATA
This converts the .json file (in the visual instruction tuning data format) to .tsv file (that the VLMEvalKit accepts)

```Shell
cd $PROJECT_DIR/VLMEvalKit
python -m vlmeval.build_our_data
```

# TRAIN / EVALUATE ON SYNTHETIC DATA
fill out cluster-specific details in bash script

```Shell
cd $PROJECT_DIR
source scripts/launcher.sh # see the script for example usage
```
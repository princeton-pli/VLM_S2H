### Prepare Conda Environment

Prepare a conda environment for the Qwen scripts using the following commands
```Shell
cd $PROJECT_DIR
conda create -n VLM_S2H_qwen python=3.10 -y
conda activate VLM_S2H_qwen
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.49.0
pip install git+https://github.com/huggingface/transformers.git@8ee50537fe7613b87881cd043a85971c85e99519
pip install trl==0.13.0
pip install datasets==3.0.2
pip install bitsandbytes==0.44.1, peft==0.13.2, qwen-vl-utils==0.0.8, wandb==0.18.5, accelerate==1.4.0
pip install flash-attn==2.7.2.post1
pip install validators==0.28.3
pip install matplotlib==3.9.2
pip install seaborn==0.13.2
pip install tabulate==0.9.0
pip install sty==1.0.6
pip install anthropic==0.32.0
pip install portalocker==2.10.1
pip install pycocoevalcap==1.2
pip install omegaconf==2.3.0
pip install timm==0.9.11
pip install fvcore==0.1.5.post20221221
pip install scipy==1.14.1
pip install XlsxWriter==3.2.0
pip install openpyxl==3.1.4
pip install -e VLMEvalKit # requirement: pip < 25.0
```

### Notes
1. Try using `fsdp_config_offload.yaml` instead if you run out of memory
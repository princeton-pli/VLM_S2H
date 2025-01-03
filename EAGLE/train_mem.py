try:
    import transformer_engine
    import transformer_engine_torch
except:
    print("having trouble importing transformer-engine!")
    
from train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

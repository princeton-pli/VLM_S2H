import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from trl import SFTConfig, SFTTrainer
import json
import numpy as np
from datasets import Dataset
from PIL import Image
from io import BytesIO
import base64
import argparse
import datasets
from torch.utils.data import DataLoader, SequentialSampler
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from accelerate import Accelerator

class SequentialSFTTrainer(SFTTrainer):

    ## modified from transformers.trainer
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = SequentialSampler(self.train_dataset) ## changed to sequential instead of random
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    



CHAT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n|im_start|>assistant\n{}<|im_end|>\n"

def encode_image_to_base64(data):
    img = Image.fromarray(data)
    im_file = BytesIO()
    img.save(im_file, format="jpeg")
    im_bytes = im_file.getvalue()
    base64_bytes = base64.b64encode(im_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

def has_image(example):
    return "image_RGB" in example and len(example["image_RGB"]) > 0

def apply_chat_template(example):
    user_prompt = example["conversations"][0]["value"]
    user_prompt = user_prompt.replace("<image>\n\n", "")
    user_prompt = user_prompt.replace("<image>\n", "")
    user_prompt = user_prompt.replace("<image>", "")
    assistant_prompt = example["conversations"][1]["value"]

    if has_image(example):
        user_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + user_prompt

    return CHAT_TEMPLATE.format(user_prompt, assistant_prompt)

def extract_image_inputs(example):
    if has_image(example):
        data = np.load(example["image_RGB"])
        data = data[example["image_index"]]
    else:
        data = np.zeros((100, 100, 3), dtype=np.uint8) ## dummy image / will be removed 
    
    base64_string = encode_image_to_base64(data)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "data:image;base64,{}".format(base64_string),
                },
            ],
        }
    ]
    image_inputs, _ = process_vision_info(messages)

    return image_inputs[0]

def reorder_by_boolean(examples, example_has_image):
    left, right = 0, 0
    while right < len(examples):
        if example_has_image[right]:  # If the flag is True, swap it forward
            examples[left], examples[right] = examples[right], examples[left]
            left += 1
        right += 1

## adapted from
## https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl
def collate_fn(examples):
    example_has_image = [has_image(example) for example in examples]
    texts = [apply_chat_template(example) for example in examples]

    if np.any(example_has_image):
        reorder_by_boolean(examples, example_has_image)
        image_inputs = [extract_image_inputs(example) for example in examples]
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )

        ## discard the pixel values of the dummy image (text-only data)
        ## and only retain the pixel values of the (image, text) pair data
        example_has_image = [has_image(example) for example in examples]
        num_pixels = [torch.prod(x).item() for x in batch['image_grid_thw']]
        pixel_mask = [b for b, n in zip(example_has_image, num_pixels) for _ in range(n)]
        batch['pixel_values'] = batch['pixel_values'][pixel_mask]
        batch['image_grid_thw'] = batch['image_grid_thw'][example_has_image]
    else:
        batch = processor(
            text=texts, return_tensors="pt", padding=True
        )
        
    labels = batch["input_ids"].clone()
    ignore_tokens = [151652, 151653, 151655, processor.tokenizer.pad_token_id] # image_tokens and pad_token
    for ignore_token_id in ignore_tokens:
        labels[labels == ignore_token_id] = -100 
    batch["labels"] = labels
    return batch

def load_dataset(data_path):
    with open(data_path) as f:
        data = json.load(f)
    data_dict = {
        "conversations": [d["conversations"] for d in data],
        "image_RGB": [d["image_RGB"][0] if len(d["image_RGB"]) > 0 else "" for d in data],
        "image_index": [d["image_RGB"][1] if len(d["image_RGB"]) > 0 else 0 for d in data],
    }
    dataset = Dataset.from_dict(data_dict)
    return dataset

def main(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_torch_fused",  # faster implementation of AdamW
        weight_decay=0.0, 
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03, 
        logging_steps=1,
        bf16=True,
        tf32=True,
        report_to="wandb",
        gradient_checkpointing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False
    )

    train_dataset = load_dataset(args.data_path)

    trainer = SequentialSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="../checkpoints")
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=20000)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    if args.save_total_limit == 0:
        args.save_total_limit = None

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, min_pixels=128*28*28, max_pixels=1280*28*28)

    main(args)
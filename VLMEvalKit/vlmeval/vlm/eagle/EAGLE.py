import torch
from PIL import Image

import sys
import os.path as osp

from ..base import BaseModel
from ...smp import *

import warnings

from .constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .conversation import conv_templates, SeparatorStyle
from .model.builder import load_pretrained_model
from .utils import disable_torch_init
from .mm_utils import tokenizer_image_token, tokenizer_image_token_llama3, get_model_name_from_path, process_images, KeywordsStoppingCriteria


class EAGLE(BaseModel):

    INTERLEAVE = False

    def __init__(self,
                 model_pth='liuhaotian/llava_v1.5_7b',
                 **kwargs):
        assert model_pth is not None
        # try:
        
        # except:
        #     warnings.warn('Please install eagle before using EAGLE')
        #     sys.exit(-1)

        print(model_pth)
        assert osp.exists(model_pth) or splitlen(model_pth) == 2

        model_name = get_model_name_from_path(model_pth)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_pth, 
            None, 
            model_name, 
            False, 
            False)

        self.model = self.model.cuda()
        self.conv_mode = "llava_llama_3"
        self.conv_templates = conv_templates


        self.tokenizer_image_token_llama3 = tokenizer_image_token_llama3
        self.process_images = process_images

    def process(self, image, question):
        if self.model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if image is None:
            image_size = None
            image_tensor = None
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.cuda()
        else:
            image_size = [image.size]
            image_tensor = self.process_images([image], self.image_processor, self.model.config)
            input_ids = self.tokenizer_image_token_llama3(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.unsqueeze(0).cuda()
            image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

        return input_ids, image_tensor, image_size, prompt

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        if image_path is None:
            image = None
        else:
            image = Image.open(image_path).convert('RGB')
        input_ids, image_tensor, image_sizes, prompt = self.process(image, prompt)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                num_beams=1,
                max_new_tokens=2048,
                use_cache=True
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
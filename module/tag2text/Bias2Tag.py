'''
 * The Tag2Text Model
 * Written by Xinyu Huang
 * Edited by Jungwook Seo
'''

import os
import json
import glob
import gc
import itertools

import torch
from tqdm import tqdm

from PIL import Image
from .ram.models import tag2text
from .ram import inference_tag2text as inference
from .ram import get_transform
        
        
class Bias2Tag():
    def __init__(self, 
                 gpu_num: int, 
                 dataset: str, 
                 user_label: dict[str: str],
                 conflict_ratio: str, 
                 root_path: str,
                 pretrained_path: str,
                 tag2text_thres: float,
                 image_size=224):
        self.root_path = root_path
        self.pretrainted_path = pretrained_path
        self.conflict_ratio = conflict_ratio
        self.dataset = dataset
        self.pretrained_path = os.path.join(pretrained_path, 'tag2text', 'tag2text_swin_14m.pth')
        self.image_size = image_size
        self.tag2text_thres = tag2text_thres
        self.device = torch.device(f'cuda:{str(gpu_num)}' if torch.cuda.is_available() else 'cpu')
        self.tag2text_model = None
        self.user_label = user_label

    def load_model(self):
        self.tag2text_model = tag2text(pretrained=self.pretrained_path,
                                       image_size=self.image_size,
                                       vit='swin_b')
        self.tag2text_model.thres = self.tag2text_thres  # thres for tagging
        self.tag2text_model.eval()
        self.tag2text_model = self.tag2text_model.to(self.device)
        print(f"Tag2Text has been loaded. Device: {self.device}")

    def off_model(self):
        del self.tag2text_model
        torch.cuda.empty_cache()
        gc.collect()
        self.tag2text_model = None

    def generate_tag2text_json(self):
        # Load tag2text.
        if self.tag2text_model == None: self.load_model()

        # Generate tag2text.json.
        transform = get_transform(dataset=self.dataset,
                                  image_size=self.image_size)
        
        # For each class and align/conflict, generate tag2text.json.
        for label, bias in itertools.product(self.user_label, ['align', 'conflict']):
            image_paths = glob.glob(os.path.join(self.root_path, 'benchmarks', self.dataset, self.conflict_ratio+'pct', bias, label, '*.png'))

            # Note that we do not use bias attribute during debiasing.
            json_dict = {}
            for image_path in tqdm(image_paths, desc=f"tag2text.json... | label: {label}, bias: {bias}"):
                bias_label = image_path.split('/')[-1].split('_')[-1][0]
                image_id = image_path.split('/')[-1] # *.png
                
                # Inference tags and caption.
                image = transform(Image.open(image_path)).unsqueeze(0).to(self.device)
                res = inference(image, self.tag2text_model)

                # Append new data to json_dict.
                # json_dict is tag2text.json dictionary.
                json_dict[image_id] = {
                    "user_label": self.user_label[label],
                    "label": label,
                    "bias_label": bias_label,
                    "biased": True if label == bias_label else False,
                    "tags": {key: None for key in res[0].split(' | ')}, # not to modify official Tag2Text code.
                    "caption": res[2],
                    "tag2text_thres": self.tag2text_thres
                }
                
            # Save /{class}/{bias}/tag2text.json
            save_json_path = os.path.join(self.root_path, 'preprocessed', self.dataset, self.conflict_ratio+'pct', bias, label, 'jsons', 'tag2text.json')
            with open(save_json_path, 'w') as file:
                json.dump(json_dict, file, indent=4)
                
        # Load off model.
        self.off_model()
        
        print("[Done] Tag2Text: tag2text.json files have been made.")        
import os
import torch
import json
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import itertools
import gc

class IP2P:
    def __init__(self, 
                 gpu_num: int, 
                 dataset: str, 
                 user_label: dict[str: str],
                 conflict_ratio: str, 
                 root_path: str,
                 pretrained_path: str,
                 random_seed: int,
                 resize_in: int=512):
        self.root_path = root_path
        self.pretrained_path = os.path.join(pretrained_path, 'InstructPix2Pix')
        self.conflict_ratio = conflict_ratio
        self.dataset = dataset
        self.device = torch.device(f'cuda:{str(gpu_num)}' if torch.cuda.is_available() else 'cpu')
        self.pipe = None
        self.generator = torch.Generator(self.device).manual_seed(random_seed)
        self.user_label = user_label
        self.resize_in = resize_in
        
    def load_model(self):
        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,
                                                                           torch_dtype=torch.float16, 
                                                                           safety_checker=None,
                                                                           cache_dir=self.pretrained_path)
        self.pipe.to(self.device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
    def off_model(self):
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        self.pip = None
        
    def resize_image(self, inout: str, image_path: str=None, image: Image.Image=None):
        if inout == 'in':
            image = Image.open(image_path)
            resized_image = image.resize((self.resize_in, self.resize_in), Image.BICUBIC)
            return resized_image
        elif inout == 'out':
            if self.dataset == 'cmnist':
                resized_image = image.resize((28, 28), Image.BICUBIC)
            elif self.dataset == 'bffhq':
                resized_image = image.resize((224, 224), Image.BICUBIC)
            return resized_image
    
    def return_insts(self, user_label: str, conflict_pairs: dict[str: list[str]]):
        prompts = []
        for bias in conflict_pairs:
            for conflict in conflict_pairs[bias]:
                prompts.append(f"Make {user_label} into {user_label} {conflict}.")
        return prompts
    
    def edit_images(self):
        if self.pipe == None: self.load_model()
        
        for label, bias in itertools.product(self.user_label, ['align', 'conflict']):
            # Load conflict.json
            conflict_json_path = os.path.join(self.root_path, 
                                              'preprocessed', 
                                              self.dataset, 
                                              self.conflict_ratio+'pct', 
                                              bias, label, 'jsons', 
                                              'conflict.json')
            if os.path.exists(conflict_json_path):
                with open(conflict_json_path, 'r') as file:
                    try:
                        conflict_json = json.load(file)
                    except json.JSONDecodeError:
                        raise RuntimeError("An error occurred while loading the existing json file.")
            else:
                raise RuntimeError(f"conflict.json does not exist.\nPath: {conflict_json_path}")
            
            for image_id in conflict_json:
                origin_image_path = os.path.join(self.root_path, 
                                                 'benchmarks', 
                                                 self.dataset, 
                                                 self.conflict_ratio+'pct', 
                                                 bias, label, image_id)
                input_image = self.resize_image(inout='in', image_path=origin_image_path)
                input_insts = self.return_insts(user_label=conflict_json[image_id]['user_label'],
                                                conflict_pairs=conflict_json[image_id]['conflict_pairs'])
                for inst in input_insts:
                    images = self.pipe(inst, image=input_image, 
                                       num_inference_steps=10, 
                                       image_guidance_scale=1, 
                                       generator=self.generator).images
                    save_path = os.path.join(self.root_path,
                                             'preprocessed', 
                                             self.dataset, 
                                             self.conflict_ratio+'pct', 
                                             bias, label, 'imgs', 
                                             f"{inst.replace(' ', '-')}_{image_id}")
                    out_image = self.resize_image(inout='out', image=images[0])
                    out_image.save(save_path, format='PNG')
                    
            print(f"[Done] IP2P: Images have been edited. | label: {label} bias: {bias}")
            
        self.off_model()
        print(f"[Done] IP2P: Images have been edited. | Whole dataset")
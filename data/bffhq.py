from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import glob
import os
import torch
import random
    
    
class bFFHQDataset(Dataset):
    def __init__(self,
                 args,
                 split: str, 
                 conflict_ratio: str,
                 root_path: str,
                 with_edited: bool,
                 class_name: dict,
                 tag_stats: dict):
        super().__init__()
        self.args = args
        self.class_name = class_name
        self.tag_stats = tag_stats
        self.transform = {
            "train": T.Compose([
                T.Resize((224,224)),
                T.RandomCrop(224, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]),
            "valid": T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]),
            "test": T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            }
        self.split = split
        self.conflict_ratio = conflict_ratio
        self.root_path = root_path
        
        if split=='train':
            # Benchmark cmnist
            self.align = glob.glob(os.path.join(root_path, 'benchmarks', 'bffhq', conflict_ratio+'pct', 'align', '*', '*')) # for each *, class_idx and image_id
            self.conflict = glob.glob(os.path.join(root_path, 'benchmarks', 'bffhq', conflict_ratio+'pct', 'conflict', '*', '*'))

            # Edited cmnist
            self.edited_align = glob.glob(os.path.join(root_path, self.args.preproc, 'bffhq', conflict_ratio+'pct', 'align', '*', 'imgs', '*')) # for each *, class_idx and image_id
            self.edited_conflict = glob.glob(os.path.join(root_path, self.args.preproc, 'bffhq', conflict_ratio+'pct', 'conflict', '*', 'imgs', '*'))
            
            if with_edited and self.args.train_method in ['with_edited', 'lff']:
                self.data = self.align + self.conflict
                self.data += self.edited_align + self.edited_conflict
            if self.args.train_method == 'naive':
                self.data = self.align + self.conflict
            if self.args.train_method == 'mixup':
                self.data = self.get_pairs()
            
        elif split=='valid':
            # Benchmark cmnist
            self.data = glob.glob(os.path.join(root_path, 'benchmarks', 'bffhq', 'valid', '*')) # *: image_id

        elif split=='test':
            # Benchmark cmnist
            self.data = glob.glob(os.path.join(root_path, 'benchmarks', 'bffhq', 'test', '*')) # *: image_id
            
        else:
            raise KeyError("Choose one of the three splits: train, valid, test")
        
    def get_insts(self, class_idx):
        bias_conflict_tags = self.tag_stats[class_idx].get("bias_conflict_tags", [])
        
        insts = []
        for bias_conflict_tag in bias_conflict_tags:
            inst = f"Turn-{self.class_name[class_idx]}-into-{self.class_name[class_idx]}-{bias_conflict_tag}".replace(' ', '-')
            insts.append(inst)
    
        return insts

    def get_pairs(self):
        pairs = []
        # align에 대한 pair
        for path in self.align:
            pair_paths = []
            origin_image_id = os.path.basename(path)
            image_idx = origin_image_id.split('_')[-3]
            class_idx = origin_image_id.split('_')[-2]
            bias_idx = origin_image_id.split('_')[-1].split('.')[0]
            insts = self.get_insts(class_idx)
            for inst in insts:    
                image_id = f'{inst}_{image_idx}_{class_idx}_{bias_idx}'
                pair_paths.append(os.path.join(self.root_path, self.args.preproc, 'bffhq', self.conflict_ratio + 'pct', 'align', class_idx, 'imgs', image_id+'.png'))
            pairs.append([path, pair_paths])

        #conflict에 대한 pair
        for path in self.conflict:
            pair_paths = []
            origin_image_id = os.path.basename(path)
            image_idx = origin_image_id.split('_')[-3]
            class_idx = origin_image_id.split('_')[-2]
            bias_idx = origin_image_id.split('_')[-1].split('.')[0]
            insts = self.get_insts(class_idx)
            for inst in insts:    
                image_id = f'{inst}_{image_idx}_{class_idx}_{bias_idx}'
                pair_paths.append(os.path.join(self.root_path, self.args.preproc, 'bffhq', self.conflict_ratio + 'pct', 'conflict', class_idx, 'imgs', image_id, '.png'))
            pairs.append([path, pair_paths])

        return pairs
        
    def mix_up(self, original_img_path, edited_img_paths):
        original_img = self.transform[self.split](Image.open(original_img_path).convert('RGB'))
        exist_edited_img_paths = [path for path in edited_img_paths if os.path.exists(path)]
        
        if exist_edited_img_paths:
            lambda_ = torch.rand(1)
            original_ratio = torch.max(lambda_, 1-lambda_)
            remaining_ratio = 1 - original_ratio
            
            edited_imgs = torch.stack([
                self.transform[self.split](Image.open(path).convert('RGB')) for path in exist_edited_img_paths
            ])
            
            edited_ratios = torch.rand(edited_imgs.size(0))
            edited_ratios = (edited_ratios / edited_ratios.sum()) * remaining_ratio
            
            mixed_image = original_img * original_ratio + (edited_imgs * edited_ratios.view(-1, 1, 1, 1)).sum(dim=0)
            
            return mixed_image
        else:
            return original_img
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.args.train_method == 'mixup' and self.split == 'train':
            original_img_path, edited_imgs_path = self.data[index]
            mixed_image = self.mix_up(original_img_path, edited_imgs_path)
            class_idx = torch.tensor(int(original_img_path.split('_')[-2]))
            bias_idx = torch.tensor(int(original_img_path.split('_')[-1].split('.')[0]))

            return mixed_image, class_idx, bias_idx, original_img_path
        
        elif self.args.train_method in ['naive', 'with_edited'] or self.split != 'train':
            class_idx = torch.tensor(int(self.data[index].split('_')[-2]))
            bias_idx = torch.tensor(int(self.data[index].split('_')[-1].split('.')[0]))
            image = Image.open(self.data[index]).convert('RGB')
            image = self.transform[self.split](image)
            
            return image, class_idx, bias_idx, self.data[index]
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import glob
import os
import torch
import random

class BARDataset(Dataset):
    def __init__(self,
                 args,
                 split: str, 
                 conflict_ratio: str,
                 root_path: str,
                 with_edited: bool,
                 class_name: dict,
                 tag_stats: dict):
        super(BARDataset, self).__init__()
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
            "before_mixup_train": T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]),
            "after_mixup_train": T.Compose([
                T.RandomCrop(224, padding=4),
                T.RandomHorizontalFlip(),
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
            self.align = glob.glob(os.path.join(root_path, 'benchmarks', 'bar', conflict_ratio+'pct', 'align', '*', '*')) # for each *, class_idx and image_id
            self.conflict = glob.glob(os.path.join(root_path, 'benchmarks', 'bar', conflict_ratio+'pct', 'conflict', '*', '*'))
            self.data = self.align + self.conflict
        
        elif split=='valid':
            # Benchmark cmnist
            self.data = glob.glob(os.path.join(root_path, 'benchmarks', 'bar', 'valid', '*')) # *: image_id

        elif split=='test':
            # Benchmark cmnist
            self.data = glob.glob(os.path.join(root_path, 'benchmarks', 'bar', 'test', '*')) # *: image_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # attr = torch.LongTensor(
        #     [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        # image = Image.open(self.data[index]).convert('RGB')
        # image_path = self.data[index]

        # if 'bar/train/conflict' in image_path:
        #     attr[1] = (attr[0] + 1) % 6
        # elif 'bar/train/align' in image_path:
        #     attr[1] = attr[0]

        # if self.transform is not None:
        #     image = self.transform(image)  
        # return image, attr, (image_path, index)
    
        class_idx = torch.tensor(int(self.data[index].split('_')[-2]))
        bias_idx = torch.tensor(int(self.data[index].split('_')[-1].split('.')[0]))
        image = Image.open(self.data[index]).convert('RGB')
        image = self.transform[self.split](image)
        
        return image, class_idx, bias_idx, self.data[index]
    
    

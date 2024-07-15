from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import glob
import os
import torch
    
    
class bFFHQDataset(Dataset):
    def __init__(self, 
                 split: str, 
                 conflict_ratio: str,
                 root_path: str,
                 with_edited: bool,
                 train_method: str):
        super().__init__()
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
            self.align = glob.glob(os.path.join(root_path, 'benchmarks', 'bffhq', conflict_ratio+'pct', 'align', '*', '*')) # for each *, label and image_id
            self.conflict = glob.glob(os.path.join(root_path, 'benchmarks', 'bffhq', conflict_ratio+'pct', 'conflict', '*', '*'))

            # Edited cmnist
            self.edited_align = glob.glob(os.path.join(root_path, 'preprocessed', 'bffhq', conflict_ratio+'pct', 'align', '*', 'imgs', '*')) # for each *, label and image_id
            self.edited_conflict = glob.glob(os.path.join(root_path, 'preprocessed', 'bffhq', conflict_ratio+'pct', 'conflict', '*', 'imgs', '*'))
            
            self.data = self.align + self.conflict
            if with_edited and train_method not in ['pairing']:
                self.data += self.edited_align + self.edited_conflict
            
        elif split=='valid':
            # Benchmark cmnist
            self.data = glob.glob(os.path.join(root_path, 'benchmarks', 'bffhq', 'valid', '*')) # *: image_id

        elif split=='test':
            # Benchmark cmnist
            self.data = glob.glob(os.path.join(root_path, 'benchmarks', 'bffhq', 'test', '*')) # *: image_id
            
        else:
            raise KeyError("Choose one of the three splits: train, valid, test")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = torch.tensor(int(self.data[index].split('_')[-2]))
        bias = torch.tensor(int(self.data[index].split('_')[-1].split('.')[0]))
        image = Image.open(self.data[index]).convert('RGB')
        image = self.transform[self.split](image)
        
        return image, label, bias, self.data[index]
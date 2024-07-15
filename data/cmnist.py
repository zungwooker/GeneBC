from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import glob
import os
import torch
    
    
class CMNISTDataset(Dataset):
    def __init__(self, 
                 split: str, 
                 conflict_ratio: str, 
                 root_path: str,
                 with_edited: bool,
                 train_method: str):
        super().__init__()
        self.transform = {
            "train": T.Compose([T.ToTensor()]),
            "valid": T.Compose([T.ToTensor()]),
            "test": T.Compose([T.ToTensor()]),
            }
        self.split = split
        self.conflict_ratio = conflict_ratio
        self.root_path = root_path
        
        if split=='train':
            # Benchmark cmnist
            self.align = glob.glob(os.path.join(root_path, 'benchmarks', 'cmnist', conflict_ratio+'pct', 'align', '*', '*')) # for each *, label and image_id
            self.conflict = glob.glob(os.path.join(root_path, 'benchmarks', 'cmnist', conflict_ratio+'pct', 'conflict', '*', '*'))

            # Edited cmnist
            self.edited_align = glob.glob(os.path.join(root_path, 'preprocessed', 'cmnist', conflict_ratio+'pct', 'align', '*', 'imgs', '*')) # for each *, label and image_id
            self.edited_conflict = glob.glob(os.path.join(root_path, 'preprocessed', 'cmnist', conflict_ratio+'pct', 'conflict', '*', 'imgs', '*'))
            
            self.data = self.align + self.conflict
            if with_edited and train_method not in ['pairing']:
                self.data += self.edited_align + self.edited_conflict
            
        elif split=='valid':
            # Benchmark cmnist
            self.data = glob.glob(os.path.join(root_path, 'benchmarks', 'cmnist', conflict_ratio+'pct', 'valid', '*')) # *: image_id

        elif split=='test':
            # Benchmark cmnist
            self.data = glob.glob(os.path.join(root_path, 'benchmarks', 'cmnist', 'test', '*', '*')) # for each *, label and image_id
            
        else:
            raise KeyError("Choose one of the three splits: train, valid, test")
        
        
        if train_method in ['pairing']:
            print("[Working] Preparing CMNIST pair dataset.")
            self.pair_data = {}
            for datum in self.data:
                self.pair_data[datum] = glob.glob()

        # FIXME

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = torch.tensor(int(self.data[index].split('_')[-2]))
        bias = torch.tensor(int(self.data[index].split('_')[-1].split('.')[0]))
        image = Image.open(self.data[index]).convert('RGB')
        image = self.transform[self.split](image)
        
        return image, label, bias, self.data[index]
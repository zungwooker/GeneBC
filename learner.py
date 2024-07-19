from pathlib import Path
import os
import glob
from itertools import chain
import torch
import torch.utils
import torch.utils.data
from rich.progress import track
import wandb
import itertools
from concurrent.futures import ThreadPoolExecutor

from module import get_model, GeneralizedCELoss
from data import get_dataset
from utils import fix
from tqdm import tqdm

class Learner():
    def __init__(self, args) -> None:
        self.args = args
        self.datasets = None
        self.dataloaders = None
        self.models = None
        self.optims = None
        self.criterions = None
        self.device = torch.device(f'cuda:{str(args.gpu_num)}' if torch.cuda.is_available() else 'cpu')
        
        
    def prepare(self):
        # Fix random seed
        fix(self.args.seed)
        
        # Datasets & Dataloaders
        self.datasets = {
            'train': get_dataset(args=self.args,
                                 dataset=self.args.dataset,
                                 split='train',
                                 conflict_ratio=self.args.conflict_ratio,
                                 with_edited=self.args.with_edited,
                                 root_path=self.args.root_path),                                  
            'valid': get_dataset(args=self.args,
                                 dataset=self.args.dataset,
                                 split='valid',
                                 conflict_ratio=self.args.conflict_ratio,
                                 root_path=self.args.root_path),            
            'test': get_dataset(args=self.args,
                                dataset=self.args.dataset,
                                split='test',
                                root_path=self.args.root_path)
            }
        
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(dataset=self.datasets['train'],
                                                 batch_size=self.args.batch_size,
                                                 shuffle=True,
                                                 num_workers=4),
            'valid': torch.utils.data.DataLoader(dataset=self.datasets['valid'],
                                                 batch_size=self.args.batch_size,
                                                 shuffle=False,
                                                 num_workers=4),
            'test': torch.utils.data.DataLoader(dataset=self.datasets['test'],
                                                batch_size=self.args.batch_size,
                                                shuffle=False,
                                                num_workers=4)
            }
        
        # Models
        self.models = {
            'biased': get_model(dataset=self.args.dataset),
            'debiased': get_model(dataset=self.args.dataset)
            }
        
        # Optimizers
        self.optims = {
            'biased': torch.optim.Adam(params=self.models['biased'].parameters(), lr=self.args.lr),
            'debiased': torch.optim.Adam(params=self.models['debiased'].parameters(), lr=self.args.lr)
            }
        
        # Criterions
        self.criterions = {
            'CELoss': torch.nn.CrossEntropyLoss(reduction='none'),
            'GCELoss': GeneralizedCELoss()
            }
        
        
    def save_model(self, model_name, save_name):
        self.models[model_name].eval()
        save_path = os.path.join(Path(__file__).parent, 'trainedModels', self.args.projcode, self.args.run_name)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.models[model_name], save_path+f'/{save_name}')
            
            
    def lff_train(self, epoch):
        self.models['biased'] = self.models['biased'].to(self.device)
        self.models['debiased'] = self.models['debiased'].to(self.device)
        self.models['biased'].train()
        self.models['debiased'].train()
        
        for batch_idx, (X, y, *_) in track(enumerate(self.dataloaders['train']), description=f'Train | epoch {epoch}...'):
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward: difficulty score
            with torch.no_grad():
                logits_b, logits_d = self.models['biased'](X), self.models['debiased'](X)
                CELoss_b, CELoss_d = self.criterions['CELoss'](logits_b, y), self.criterions['CELoss'](logits_d, y)
                diff_score = (CELoss_b+1e-10)/(CELoss_b+CELoss_d+1e-10)
                
            # Forward: Losses
            logits_b, logits_d = self.models['biased'](X), self.models['debiased'](X)
            GCELoss_b, CELoss_d = self.criterions['GCELoss'](logits_b, y), self.criterions['CELoss'](logits_d, y)
            
            # Total loss configuration
            total_loss = (GCELoss_b + CELoss_d * diff_score).mean()
            
            # Update
            self.optims['biased'].zero_grad()
            self.optims['debiased'].zero_grad()
            total_loss.backward()
            self.optims['biased'].step()
            self.optims['debiased'].step()
            
            if self.args.wandb:
                wandb.log({
                    "Iter step": (batch_idx+1) + epoch*len(self.dataloaders['train']),
                    "training/total loss": total_loss,
                    "training/B: GCELoss": GCELoss_b.mean(),
                    "training/B: CELoss": CELoss_b.mean(),
                    "training/D: CELoss": CELoss_d.mean(),
                    "training/D: CELoss * diff_score": (CELoss_d * diff_score).mean(),
                })


    def train(self, epoch):
        self.models['debiased'] = self.models['debiased'].to(self.device)
        self.models['debiased'].train()
        
        for batch_idx, (X, y, *_) in track(enumerate(self.dataloaders['train']), description=f'Train | epoch {epoch}...', total=len(self.dataloaders['train'])):
            X, y = X.to(self.device), y.to(self.device)
                
            # Forward: Losses
            logits_d = self.models['debiased'](X)
            CELoss_d = self.criterions['CELoss'](logits_d, y)
            
            # Total loss configuration
            total_loss =  CELoss_d.mean()
            
            # Update
            self.optims['debiased'].zero_grad()
            total_loss.backward()
            self.optims['debiased'].step()
            
            if self.args.wandb:
                wandb.log({
                    "Iter step": (batch_idx+1) + epoch*len(self.dataloaders['train']),
                    "training/D: CELoss": CELoss_d.mean(),
                })
                              

    # Work in progress
    def pairing_train(self, epoch):
        self.models['debiased'] = self.models['debiased'].to(self.device)
        self.models['debiased'].train()
        
        def find_image_paths(path, image_id):
            context_path = path.replace('benchmarks', self.args.preproc).replace(f'{image_id}', f'imgs/*_{image_id}')
            return glob.glob(context_path)
        
        for batch_idx, (X, y, *_) in track(enumerate(self.dataloaders['train']), description=f'Train | epoch {epoch}...'):
            X, y = X.to(self.device), y.to(self.device)
            
            # Configure batch
            image_ids = [os.path.basename(image) for image in path]
            # pair_image_paths = []
            # for path_idx in range(len(path)):
            #     context_path = path[path_idx].replace('benchmarks', self.args.preproc).replace(f'{image_ids[path_idx]}', f'imgs/*_{image_ids[path_idx]}')
            #     pair_image_paths += glob.glob(context_path)
            with ThreadPoolExecutor() as executor:
                pair_image_paths = list(chain.from_iterable(executor.map(find_image_paths, path, image_ids)))
            breakpoint()

            # Forward: Losses
            logits_d = self.models['debiased'](X)
            CELoss_d = self.criterions['CELoss'](logits_d, y)
            
            # Total loss configuration
            total_loss =  CELoss_d.mean()
            
            # Update
            self.optims['debiased'].zero_grad()
            total_loss.backward()
            self.optims['debiased'].step()
            
            if self.args.wandb:
                wandb.log({
                    "Iter step": (batch_idx+1) + epoch*len(self.dataloaders['train']),
                    "training/D: CELoss": CELoss_d.mean(),
                })
        
        
    # def eval(self, model_name):
    #     with torch.no_grad():
    #         self.models[model_name] = self.models[model_name].to(self.device)
    #         self.models[model_name].eval()
            
    #         splits = ['train', 'valid', 'test']
    #         bias_attributes = ['whole', 'aligned', 'conflict']
    #         metrics = {
    #             'train': {},
    #             'valid': {},
    #             'test': {},
    #         }
    #         for split, bias_attribute in itertools.product(splits, bias_attributes):
    #             metrics[split][bias_attribute] = {
    #                 'total_loss': 0,
    #                 'loss': None,
    #                 'correct': 0,
    #                 'total_num': 0,
    #                 'accuracy': None,
    #             }
    #             for batch_idx, (X, y, bias, *_) in track(enumerate(self.dataloaders[split]), description=f'Eval | split {split}, bias_attr {bias_attribute} ...', total=len(self.dataloaders[split])):
    #                 X, y, bias = X.to(self.device), y.to(self.device), bias.to(self.device)
                    
    #                 if bias_attribute == 'aligned':
    #                     X, y = X[y==bias], y[y==bias]
    #                 if bias_attribute == 'conflict':
    #                     X, y = X[y!=bias], y[y!=bias]

    #                 # Case no B.C. in this batch.
    #                 if X.size(0) == 0: continue
                        
    #                 # Forward
    #                 logits = self.models[model_name](X)
    #                 CELoss = self.criterions['CELoss'](logits, y)
                    
    #                 # Calculate correct
    #                 _, pred_labels = torch.max(logits, dim=1)
    #                 correct = (pred_labels == y).sum()
                    
    #                 # Logging
    #                 metrics[split][bias_attribute]['total_loss'] += CELoss.sum()
    #                 metrics[split][bias_attribute]['total_num'] += y.size(0)
    #                 metrics[split][bias_attribute]['correct'] += correct
                
    #             if metrics[split][bias_attribute]['total_num'] > 0:
    #                 metrics[split][bias_attribute]['loss'] = metrics[split][bias_attribute]['total_loss']/metrics[split][bias_attribute]['total_num']
    #                 metrics[split][bias_attribute]['accuracy'] = metrics[split][bias_attribute]['correct']/metrics[split][bias_attribute]['total_num']
    #             else:
    #                 metrics[split][bias_attribute]['loss'] = 0.0
    #                 metrics[split][bias_attribute]['accuracy'] = 0.0      
                
        
    #     self.metrics = metrics
    #     print(metrics)
        
    def evaluate(self, model, data_loader):
        model.eval()
        total_correct, total_num = 0, 0
        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct/float(total_num)
        model.train()

        return accs


    def wandb_switch(self, switch):
        if self.args.wandb and switch == 'start':
            wandb.init(
                project=self.args.projcode,
                name=f'{self.args.run_name} | seed: {self.args.seed}',
                config={
                },
                settings=wandb.Settings(start_method="fork")
            )
            wandb.define_metric("training/*", step_metric="Iter step")
            
            splits = ['train', 'valid', 'test']
            bias_attributes = ['whole', 'aligned', 'conflict']
            for split, bias_attribute in itertools.product(splits, bias_attributes):
                wandb.define_metric(f"{split}_{bias_attribute}/*", step_metric="Epoch step")
        
        elif self.args.wandb and switch == 'finish':
            wandb.finish()
                 
                 
    def wandb_log(self, epoch):
        splits = ['train', 'valid', 'test']
        bias_attributes = ['whole', 'aligned', 'conflict']
        for split, bias_attribute in itertools.product(splits, bias_attributes):
            wandb.log({
                "Epoch step": epoch,
                f"{split}_{bias_attribute}/Total Loss(epoch sum)": self.metrics[split][bias_attribute]['total_loss'],
                f"{split}_{bias_attribute}/Total Loss(sample-wise)": self.metrics[split][bias_attribute]['loss'],
                f"{split}_{bias_attribute}/Accuracy": self.metrics[split][bias_attribute]['accuracy'],
            })
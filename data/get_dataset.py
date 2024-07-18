from .cmnist import *
from .bffhq import bFFHQDataset
import json

def get_dataset(args,
                dataset: str,
                split: str,
                root_path: str,
                conflict_ratio: str='none',
                with_edited: bool=False):
    
    # Load class name
    class_name_path = os.path.join(args.root_path, 'benchmarks', args.dataset, 'class_name.json') # benchmarks/{dataset}/class_name.json
    if os.path.exists(class_name_path):
        with open(class_name_path, 'r') as file:
            try:
                class_name = json.load(file)
            except json.JSONDecodeError:
                raise RuntimeError("An error occurred while loading the existing json file.")
    else:
        raise RuntimeError(f"class_name.json does not exist.\nPath: {class_name_path}")
    
    # Load tag_stats
    if args.train_method not in ['naive']:
        tag_stats_path = os.path.join(args.root_path, 
                                    args.preproc,
                                    args.dataset,
                                    args.conflict_ratio+'pct',
                                    'tag_stats.json')
        if os.path.exists(tag_stats_path):
            with open(tag_stats_path, 'r') as file:
                try:
                    tag_stats = json.load(file)
                except json.JSONDecodeError:
                    raise RuntimeError("An error occurred while loading the existing json file.")
        else:
            raise RuntimeError(f"tag_stats.json does not exist.\nPath: {tag_stats_path}")
    else:
        tag_stats = None
    
    # Default: without edited images.
    if dataset == 'cmnist':
        return CMNISTDataset(split=split,
                             args=args,
                             conflict_ratio=conflict_ratio,
                             root_path=root_path,
                             with_edited=with_edited,
                             class_name=class_name,
                             tag_stats=tag_stats)
    elif dataset == 'bffhq':
        return bFFHQDataset(split=split,
                            args=args,
                            conflict_ratio=conflict_ratio,
                            root_path=root_path,
                            with_edited=with_edited,
                            class_name=class_name,
                            tag_stats=tag_stats)
    elif dataset == 'bar':
        raise KeyError("bar dataset class is not ready.")
    elif dataset == 'dogs_and_cats':
        raise KeyError("dogs_and_cats dataset class is not ready.")
    else:
        raise KeyError("Choose one of the four datasets: cmnist, bffhq, bar, dogs_and_cats")
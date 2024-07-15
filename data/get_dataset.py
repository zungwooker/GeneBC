from .cmnist import *
from .bffhq import bFFHQDataset

def get_dataset(dataset: str,
                split: str,
                root_path: str,
                conflict_ratio: str='none',
                with_edited: bool=False,
                train_method: str='none'):
    # Default: without edited images.
    if dataset == 'cmnist':
        return CMNISTDataset(split=split,
                             conflict_ratio=conflict_ratio,
                             root_path=root_path,
                             with_edited=with_edited,
                             train_method=train_method)
    elif dataset == 'bffhq':
        return bFFHQDataset(split=split,
                            conflict_ratio=conflict_ratio,
                            root_path=root_path,
                            with_edited=with_edited,
                            train_method=train_method)
    elif dataset == 'bar':
        raise KeyError("bar dataset class is not ready.")
    elif dataset == 'dogs_and_cats':
        raise KeyError("dogs_and_cats dataset class is not ready.")
    else:
        raise KeyError("Choose one of the four datasets: cmnist, bffhq, bar, dogs_and_cats")
import os
import itertools

def makedir_preprocessed(root_path: str, 
                         dataset: str,
                         conflict_ratio: str,
                         user_label: dict):
    for label, bias, data_type in itertools.product(user_label, ['align', 'conflict'], ['imgs', 'jsons']):
        dir_path = os.path.join(root_path, 
                                'preprocessed',
                                dataset,
                                conflict_ratio+'pct',
                                bias, label, data_type)
        os.makedirs(dir_path, exist_ok=True)
    print(f"[Done] makedir: preprocessed dir has been made.")
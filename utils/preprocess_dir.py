import os
import itertools

def makedir_preprocessed(args,
                         class_name: dict):
    for label, bias, data_type in itertools.product(class_name, ['align', 'conflict'], ['imgs', 'jsons']):
        dir_path = os.path.join(args.root_path, 
                                args.preproc,
                                args.dataset,
                                args.conflict_ratio+'pct',
                                bias, label, data_type)
        os.makedirs(dir_path, exist_ok=True)
    print(f"[Done] makedir: preprocessed dir has been made.")
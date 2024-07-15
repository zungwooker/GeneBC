import os
import glob
import json
import argparse
import itertools

def process_gate(args: argparse.ArgumentParser, file_name: str):
    if args.force: 
        print(f"[Process gate] Force process.")
        return False
    
    user_label_path = os.path.join(args.root_path, 'benchmarks', args.dataset, 'user_label.json') # benchmarks/{args.dataset}/user_label.json
    if os.path.exists(user_label_path):
        with open(user_label_path, 'r') as file:
            try:
                user_label = json.load(file)
            except json.JSONDecodeError:
                raise RuntimeError("An error occurred while loading the existing json file.")
    else:
        raise RuntimeError(f"user_label.json does not exist.\nPath: {user_label_path}")
    
    if '.json' in file_name:
        for label, bias in itertools.product(user_label, ['align', 'conflict']):
            json_path = os.path.join(args.root_path, 'preprocessed', args.dataset, args.conflict_ratio+'pct', bias, label, 'jsons', file_name)
            if not os.path.exists(json_path):
                print(f"[Process gate] File missing {json_path}")
                print(f"[Process gate] Working on {file_name}...")
                return False
    elif file_name == 'png':
        for label, bias in itertools.product(user_label, ['align', 'conflict']):
            conflict_json_path = os.path.join(args.root_path, 'preprocessed', args.dataset, args.conflict_ratio+'pct', bias, label, 'jsons', 'conflict.json')
            conflict_imgs_path = os.path.join(args.root_path, 'preprocessed', args.dataset, args.conflict_ratio+'pct', bias, label, 'imgs', '*.png')
            if os.path.exists(conflict_json_path):
                with open(conflict_json_path, 'r') as file:
                    try:
                        conflict_json = json.load(file)
                    except json.JSONDecodeError:
                        raise RuntimeError("An error occurred while loading the existing json file.")
            else:
                raise RuntimeError(f"conflict.json does not exist.\nPath: {conflict_json_path}")
            
            cnt_combi = 0
            cnt_edtied = len(glob.glob(conflict_imgs_path))
            for image_id in conflict_json:
                for bias in conflict_json[image_id]['conflict_pairs']:
                    cnt_combi += len(conflict_json[image_id]['conflict_pairs'][bias])
            if cnt_combi != len(glob.glob(conflict_imgs_path)):
                print(f"[Process gate] File missing. | Edited images")
                print(f"label: {label} bias: {bias} cnt_combi: {cnt_combi} cnt_edited: {cnt_edtied}")
                print(f"[Process gate] Working on {file_name}...")
                return False
        
    print(f"[Process gate] All files exists. {file_name}")
    return True
            
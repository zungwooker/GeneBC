import argparse
import os
import json

from module import Bias2Tag, BartLargeMnli, Gemini, IP2P
from module import generate_bias_candidates_json
from utils import process_gate, Timer, makedir_preprocessed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu_num", type=int, help="GPU number")
    parser.add_argument("-dataset", type=str, help="Dataset")
    parser.add_argument("-conflict_ratio", type=str, help="Conflict ratio")
    parser.add_argument("-n_bias", type=int, help="num of bias candidates: k")
    parser.add_argument("-tag2text_thres", type=float, help="tag2text thres", default=0.68)
    parser.add_argument("-sim_thres", type=float, help="Label filtering tag similarity thres", default=0.95)
    parser.add_argument("-root_path", type=str, help="Parent path of benchmarks & preprocessed.")
    parser.add_argument("-pretrained_path", type=str, help="Dir of pretrained weights.")
    parser.add_argument("-random_seed", type=int, help="Random seed for editing.")
    parser.add_argument("-fast_test", action="store_true", help="Fast test for validation of code.")
    parser.add_argument("-force", action="store_true", default=False, help="Force working: do not care previous json files.")
    args = parser.parse_args()
    
    # For fast testing
    if args.fast_test: args.root_path += '/fast_test'
    
    # Load user labels
    user_label_path = os.path.join(args.root_path, 'benchmarks', args.dataset, 'user_label.json') # benchmarks/{dataset}/user_label.json
    if os.path.exists(user_label_path):
        with open(user_label_path, 'r') as file:
            try:
                user_label = json.load(file)
            except json.JSONDecodeError:
                raise RuntimeError("An error occurred while loading the existing json file.")
    else:
        raise RuntimeError(f"user_label.json does not exist.\nPath: {user_label_path}")
    
    # Make preprocessed dirs.
    makedir_preprocessed(root_path=args.root_path,
                         dataset=args.dataset,
                         conflict_ratio=args.conflict_ratio,
                         user_label=user_label)
    
    # Timer
    timetabel_path = os.path.join(args.root_path, 'preprocessed', args.dataset, args.conflict_ratio+'pct', 'timetable.json')
    timer = Timer(save_path=timetabel_path)
    
    # Make tag2text.json: Done.
    if not process_gate(args=args, file_name='tag2text.json'):
        timer.start_record(task_name='tag2text')
        bias2tag = Bias2Tag(gpu_num=args.gpu_num,
                            dataset=args.dataset,
                            conflict_ratio=args.conflict_ratio,
                            user_label=user_label,
                            root_path=args.root_path,
                            pretrained_path=args.pretrained_path,
                            tag2text_thres=args.tag2text_thres)
        bias2tag.generate_tag2text_json()
        timer.end_record(task_name='tag2text')
    
    # Make filtered.json
    if not process_gate(args=args, file_name='filtered.json'):
        timer.start_record(task_name='label_filtering')
        bartLargeMnli = BartLargeMnli(gpu_num=args.gpu_num,
                                    dataset=args.dataset,
                                    conflict_ratio=args.conflict_ratio,
                                    user_label=user_label,
                                    root_path=args.root_path,
                                    pretrained_path=args.pretrained_path,
                                    sim_thres=args.sim_thres)
        bartLargeMnli.generate_filtered_json()
        timer.end_record(task_name='label_filtering')
    
    # Make tag_stats.json
    if not process_gate(args=args, file_name='tag_stats.json'):
        timer.start_record(task_name='tag_stats')
        generate_bias_candidates_json(dataset=args.dataset,
                                      user_label=user_label,
                                      conflict_ratio=args.conflict_ratio,
                                      n_bias=args.n_bias,
                                      root_path=args.root_path)
        timer.end_record(task_name='tag_stats')

    # Make conflict.json
    if not process_gate(args=args, file_name='conflict.json'):
        timer.start_record(task_name='conflict_pairs')
        gemini = Gemini(dataset=args.dataset,
                        conflict_ratio=args.conflict_ratio,
                        user_label=user_label,
                        root_path=args.root_path)
        gemini.generate_conflict_json()
        timer.end_record(task_name='conflict_pairs')

    # Generate images
    if not process_gate(args=args, file_name='png'):
        timer.start_record(task_name='generating_conflict_imgs')
        ip2p = IP2P(gpu_num=args.gpu_num,
                    dataset=args.dataset,
                    user_label=user_label,
                    conflict_ratio=args.conflict_ratio,
                    root_path=args.root_path,
                    pretrained_path=args.pretrained_path,
                    random_seed=args.random_seed)
        ip2p.edit_images()
        timer.end_record(task_name='generating_conflict_imgs')
    
if __name__ == '__main__':
    main()
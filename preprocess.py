import argparse
import os
import json
import time
from email.mime.text import MIMEText
import smtplib

from module import Bias2Tag, IP2P, TagStats
from utils import process_gate, Timer, makedir_preprocessed

def email(content: str):
    msg = MIMEText(f'{content}')
    msg['Subject'] = 'Preprocess Done.'
    msg['From'] = 'zungwooker@gmail.com'
    msg['To'] = 'zungwooker@hanyang.ac.kr'

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('zungwooker@gmail.com', 'rajm ucuz xyhp vbzf')

    server.sendmail('zungwooker@gmail.com', 'zungwooker@hanyang.ac.kr', msg.as_string())

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu_num", type=int, help="GPU number")
    parser.add_argument("-dataset", type=str, help="Dataset")
    parser.add_argument("-conflict_ratio", type=str, help="Conflict ratio")
    parser.add_argument("-n_bias", required=False, type=int, help="num of bias candidates: k")
    parser.add_argument("-root_path", type=str, help="Parent path of benchmarks & preprocessed.")
    parser.add_argument("-pretrained_path", type=str, help="Dir of pretrained weights.")
    parser.add_argument("-preproc", type=str, help="Dir of preprocessed data.")
    parser.add_argument("-random_seed", type=int, help="Random seed for editing.")
    parser.add_argument("-fast_test", action="store_true", help="Fast test for validation of code.")
    
    # Tag2Text
    parser.add_argument("-extract_tags", action="store_true")
    parser.add_argument("-tag2text_thres", type=float, help="tag2text thres", default=0.68)
    
    # Bart text classification
    parser.add_argument("-compute_tag_stats", action="store_true")
    parser.add_argument("-sim_thres", type=float, help="Label filtering tag similarity thres", default=0.95)
    
    # Instruct Pix2Pix
    parser.add_argument("-generate_gate", action="store_true")
    parser.add_argument("-image_guidance_scale", type=float, help="Image guidance scale size for instructPix2Pix")
    parser.add_argument("-resize_in", type=int, default=512, help="Input size for instructPix2Pix")
    parser.add_argument("-edit_class_idx", type=str, required=False, help="Class index for editing")

    # Email
    parser.add_argument("-email", type=str, required=False, help="contents")

    args = parser.parse_args()
    
    # For fast testing
    if args.fast_test: args.root_path += '/fast_test'
    
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
    
    # Make preprocessed dirs.
    makedir_preprocessed(args=args,
                         class_name=class_name)
    
    # Timer
    timetabel_path = os.path.join(args.root_path, args.preproc, args.dataset, args.conflict_ratio+'pct', f'timetable_{time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())}.json')
    timer = Timer(save_path=timetabel_path)
    
    # Make tag2text.json
    if args.extract_tags:
        timer.start_record(task_name='bias2tag')
        bias2tag = Bias2Tag(args=args, class_name=class_name,)
        bias2tag.generate_tag_json()
        timer.end_record(task_name='bias2tag')
    
    # Make tag_stats.json
    if args.compute_tag_stats:
        # timer.start_record(task_name='tag_stats')
        tag_stats = TagStats(args=args, class_name=class_name)
        # tag_stats.generate_tag_stats()
        # tag_stats.integrate_tag_stats()
        # tag_stats.condition_bias()
        # tag_stats.mix_bias()
        tag_stats.class_bias_stats()
        # timer.end_record(task_name='tag_stats')

    # Generate images
    if args.generate_gate:
        timer.start_record(task_name='generating_conflict_imgs')
        ip2p = IP2P(args=args, class_name=class_name)
        ip2p.edit_images()
        timer.end_record(task_name='generating_conflict_imgs')

    if hasattr(args, 'email'):
        email(content=args.email)
    
if __name__ == '__main__':
    main()
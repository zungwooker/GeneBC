import itertools
import json
import os
import torch
import gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

class TagStats():
    def __init__(self, args, class_name: dict) -> None:
        self.args = args
        self.class_name = class_name
        self.device = torch.device(f'cuda:{str(args.gpu_num)}' if torch.cuda.is_available() else 'cpu')
        
        
    def load_model(self):
        model_id = "facebook/bart-large-mnli"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                                        cache_dir=self.args.pretrained_path+'/bart').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                       cache_dir=self.args.pretrained_path+'/bart')
        self.classifier = pipeline("zero-shot-classification", 
                                   model=self.model, 
                                   tokenizer=self.tokenizer, 
                                   device=self.device)
        print(f"Bart-large-mnli classifier has been loaded. Device: {self.device}")
        
    def off_model(self):
        del self.model
        del self.tokenizer
        del self.classifier
        torch.cuda.empty_cache()
        gc.collect()
        self.model = None
        self.tokenizer = None
        self.classifier = None
    
    def sort_tags(self, tags: dict[str: int]):
        # Sort tags descending order in frequency.
        sorted_tags = sorted(tags.items(), key=lambda item:item[1], reverse=True)
        return dict(sorted_tags)
    
    def sort_tags_(self, tags: dict[str, dict[str, int]]):
        # Sort tags descending order in frequency.
        sorted_tags = sorted(tags.items(), key=lambda item:item[1]['appeared'][0], reverse=True)
        return dict(sorted_tags)
        
    def generate_tag_stats(self):
        # For each class and align/conflict, generate tag_stats.json 
        # and integrated tag_stats.json by class.
        for class_idx, bias_type in itertools.product(self.class_name, ['align', 'conflict']):
            # Base architecture of tag_stats.json
            tag_stats = {'n_data': 0, 'tags': {}}
            
            # Load tags.json files for {class_idx, bias_type}
            tags_json_path = os.path.join(self.args.root_path, self.args.preproc, self.args.dataset, self.args.conflict_ratio+'pct', bias_type, class_idx, 'jsons', 'tags.json')
            if os.path.exists(tags_json_path):
                with open(tags_json_path, 'r') as file:
                    try:
                        tags_json = json.load(file)
                    except json.JSONDecodeError:
                        raise RuntimeError("An error occurred while loading the existing json file.")
            else:
                raise RuntimeError(f"tags.json does not exist.\nPath: {tags_json_path}")
            
            # Count tags.
            for image_id in tags_json:
                tag_stats['n_data'] += 1
                for tag in tags_json[image_id]['tags']:
                    if tag in tag_stats['tags']: tag_stats['tags'][tag] += 1
                    else: tag_stats['tags'][tag] = 1
            if tag_stats['n_data'] == 0:
                raise RuntimeError(f"n_data cannot be zero.\nPath: {tags_json_path}")
                
            # Sort and get top k tags.
            tag_stats['tags'] = self.sort_tags(tag_stats['tags'])
            for tag in tag_stats['tags']:
                tag_stats['tags'][tag] = [tag_stats['tags'][tag], tag_stats['tags'][tag]/tag_stats['n_data']] # [n_tags appeared, appearance ratio]
                
            # Save json.
            save_json_path = os.path.join(self.args.root_path, self.args.preproc, self.args.dataset, self.args.conflict_ratio+'pct', bias_type, class_idx, 'jsons', 'tag_stats.json')
            with open(save_json_path, 'w') as file:
                json.dump(tag_stats, file, indent=4)
                
    def integrate_tag_stats(self):
        # Integrate jsons by class.
        self.itg_tag_stats = {}
        for class_idx in self.class_name:
            self.itg_tag_stats[class_idx] = {'n_data': 0, 'tags': {}}
            
            for bias_type in ['align', 'conflict']:
                # Load tags.json files for {class_idx, bias_type}
                tag_stats_json_path = os.path.join(self.args.root_path, self.args.preproc, self.args.dataset, self.args.conflict_ratio+'pct', bias_type, class_idx, 'jsons', 'tag_stats.json')
                if os.path.exists(tag_stats_json_path):
                    with open(tag_stats_json_path, 'r') as file:
                        try:
                            tag_stats = json.load(file)
                        except json.JSONDecodeError:
                            raise RuntimeError("An error occurred while loading the existing json file.")
                else:
                    raise RuntimeError(f"tag_stats.json does not exist.\nPath: {tag_stats_json_path}")
                
                # Combine results from align/conflict.
                self.itg_tag_stats[class_idx]['n_data'] += tag_stats['n_data']
                for tag in tag_stats['tags']:
                    if tag in self.itg_tag_stats[class_idx]['tags']:
                        self.itg_tag_stats[class_idx]['tags'][tag]['appeared'] += tag_stats['tags'][tag][0]
                    else:
                        self.itg_tag_stats[class_idx]['tags'][tag] = {'appeared': tag_stats['tags'][tag][0], 
                                                                      'cond1': None, 
                                                                      'cond2': None,
                                                                      'cond3': None}
            for tag in self.itg_tag_stats[class_idx]['tags']:
                n_appeared = self.itg_tag_stats[class_idx]['tags'][tag]['appeared']
                self.itg_tag_stats[class_idx]['tags'][tag]['appeared'] = [n_appeared, n_appeared/self.itg_tag_stats[class_idx]['n_data']]

                    
        # Save json.
        save_json_path = os.path.join(self.args.root_path, self.args.preproc, self.args.dataset, self.args.conflict_ratio+'pct', 'tag_stats.json')
        with open(save_json_path, 'w') as file:
            json.dump(self.itg_tag_stats, file, indent=4)
                    
                    
    def condition_bias(self):
        # Bias Attributes Validation
        # Condition 1: Intra-class correlation: The appearance ratio of a bias tag should be greater than {1/n_class}.
        # Condition 2: Inter-class correlation: If a specific tag is present in more than half of the classes (n_class/2), it is not considered a bias.
        #   - Note: For Condition 2, if Condition 1 is satisfied, the class is considered to possess the tag.
        # Condition 3: The bias tag should not be similar to the class name.
        # Bias attribute = Condition1 & Condition2 & Condition3
        
        # Check Condition. 1
        for class_idx in self.class_name:
            for tag in self.itg_tag_stats[class_idx]['tags']:
                appeared_ratio = self.itg_tag_stats[class_idx]['tags'][tag]['appeared'][-1]
                intra_corr_ratio = 1/len(self.class_name)
                condition_1 = 1 if appeared_ratio > intra_corr_ratio else 0
                self.itg_tag_stats[class_idx]['tags'][tag]['cond1'] = condition_1
        
        # Check Condition. 2
        for target_class_idx in self.class_name:
            for tag in self.itg_tag_stats[target_class_idx]['tags']:
                cond2_cnt = 0
                for subject_class_idx in self.class_name:
                    if tag in self.itg_tag_stats[subject_class_idx]['tags']:
                        if self.itg_tag_stats[subject_class_idx]['tags'][tag]['cond1'] == 1:
                            cond2_cnt += 1
                condition_2 = 1 if cond2_cnt <= len(self.class_name)/2 else 0
                self.itg_tag_stats[target_class_idx]['tags'][tag]['cond2'] = condition_2
                
        # Check Condition. 3
        self.load_model()
        for class_idx in self.class_name:
            tags = list(self.itg_tag_stats[class_idx]['tags'].keys())
            tags_sim = {}
            if len(tags) <= 1000:
                tmp_tags = tags
                res = self.classifier(f"{self.class_name[class_idx]}", tmp_tags, multi_label=True)
                tmp_tags_sim = {res['labels'][j]: res['scores'][j] for j in range(len(tags))}
                tags_sim.update(tmp_tags_sim)
            else:
                for i in range(len(tags)//1000):
                    tmp_tags = tags[i*1000:(i+1)*1000]
                    res = self.classifier(f"{self.class_name[class_idx]}", tmp_tags, multi_label=True)
                    tmp_tags_sim = {res['labels'][j]: res['scores'][j] for j in range(1000)}
                    tags_sim.update(tmp_tags_sim)
                tmp_tags = tags[(i+1)*1000:]
                res = self.classifier(f"{self.class_name[class_idx]}", tmp_tags, multi_label=True)
                tmp_tags_sim = {res['labels'][j]: res['scores'][j] for j in range(1000)}
                tags_sim.update(tmp_tags_sim)
            
            for tag in tags:
                condition_3 = 1 if tags_sim[tag] < self.args.sim_thres else 0
                self.itg_tag_stats[class_idx]['tags'][tag]['cond3'] = condition_3

        for class_idx in self.class_name:
            # Sort tags.  
            self.itg_tag_stats[class_idx]['tags'] = self.sort_tags_(self.itg_tag_stats[class_idx]['tags'])
            bias_tags = {}
            for tag in self.itg_tag_stats[class_idx]['tags']:
                if self.itg_tag_stats[class_idx]['tags'][tag]['cond1'] and self.itg_tag_stats[class_idx]['tags'][tag]['cond2'] and self.itg_tag_stats[class_idx]['tags'][tag]['cond3']:
                    bias_tags[tag] = self.itg_tag_stats[class_idx]['tags'][tag]
                if len(bias_tags) >= self.args.n_bias:
                    break
            self.itg_tag_stats[class_idx]['bias_tags'] = bias_tags
        
        # Save json.
        save_json_path = os.path.join(self.args.root_path, self.args.preproc, self.args.dataset, self.args.conflict_ratio+'pct', 'tag_stats.json')
        with open(save_json_path, 'w') as file:
            json.dump(self.itg_tag_stats, file, indent=4)
        
    def mix_bias(self):
        biases = set()
        for class_idx in self.class_name:
            for bias_tag in self.itg_tag_stats[class_idx]['bias_tags']:
                biases.add(bias_tag)
        biases = list(biases)
                
        # Do not use biases that have the same meaning as a specific class as bias-conflict attributes.
        bias_class_scores = self.classifier([value for value in self.class_name.values()], biases, multi_label=True)
        confusing_biases = []
        for i in range(len(self.class_name)):
            confusing_biases += [label for score, label in zip(bias_class_scores[i]['scores'], bias_class_scores[i]['labels']) if score >= self.args.sim_thres]
        confusing_biases = set(confusing_biases)
                
        for class_idx in self.class_name:
            tmp_biases = list(biases)
            for bias_tag in self.itg_tag_stats[class_idx]['bias_tags']:
                if bias_tag in tmp_biases:
                    tmp_biases.remove(bias_tag)
            
            tmp_biases = list(set(tmp_biases) - confusing_biases)
            self.itg_tag_stats[class_idx]['bias_conflict_tags'] = tmp_biases
            
        save_json_path = os.path.join(self.args.root_path, self.args.preproc, self.args.dataset, self.args.conflict_ratio+'pct', 'tag_stats.json')
        with open(save_json_path, 'w') as file:
            json.dump(self.itg_tag_stats, file, indent=4)
            
        print("[Done] Bias candidates: tag_stats.json files have been made.")
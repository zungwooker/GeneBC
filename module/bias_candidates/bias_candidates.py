import itertools
import json
import os

def generate_bias_candidates_json(dataset: str,
                             user_label: dict[str: str],
                             conflict_ratio: str,
                             n_bias: int,
                             root_path: str):
    def sort_tags(tags: dict[str: int]):
        # Sort tags descending order in frequency.
        sorted_tags = sorted(tags.items(), key=lambda item:item[1], reverse=True)
        return dict(sorted_tags)
    
    def sort_tags_(tags: dict[str, dict[str, int]]):
        # Sort tags descending order in frequency.
        sorted_tags = sorted(tags.items(), key=lambda item:item[1]['appeared'], reverse=True)
        return dict(sorted_tags)
    
    # For each class and align/conflict, generate tag_stats.json 
    # and integrated tag_stats.json by class.
    for label, bias in itertools.product(user_label, ['align', 'conflict']):
        tag_stats = {'n_data': 0, 'tags': {}}
        
        # Load filtered.json.
        filtered_json_path = os.path.join(root_path, 'preprocessed', dataset, conflict_ratio+'pct', bias, label, 'jsons', 'filtered.json')
        if os.path.exists(filtered_json_path):
            with open(filtered_json_path, 'r') as file:
                try:
                    filtered_json = json.load(file)
                except json.JSONDecodeError:
                    raise RuntimeError("An error occurred while loading the existing json file.")
        else:
            raise RuntimeError(f"filtered.json does not exist.\nPath: {filtered_json_path}")
        
        # Count tags.
        for image_id in filtered_json:
            tag_stats['n_data'] += 1
            for tag in filtered_json[image_id]['filtered_tags']:
                if tag in tag_stats['tags']: tag_stats['tags'][tag] += 1
                else: tag_stats['tags'][tag] = 1
        if tag_stats['n_data'] == 0:
            raise RuntimeError(f"n_data cannot be zero.\nPath{filtered_json_path}")
            
        # Sort and get top k tags.
        tag_stats['tags'] = sort_tags(tag_stats['tags'])
        for tag in tag_stats['tags']:
            tag_stats['tags'][tag] = [tag_stats['tags'][tag], tag_stats['tags'][tag]/tag_stats['n_data']]
        tag_stats['bias_candidates'] = dict(list(tag_stats['tags'].items())[:n_bias])
        tag_stats['n_bias'] = n_bias
            
        # Save json.
        save_json_path = os.path.join(root_path, 'preprocessed', dataset, conflict_ratio+'pct', bias, label, 'jsons', 'tag_stats.json')
        with open(save_json_path, 'w') as file:
            json.dump(tag_stats, file, indent=4)
            
    # Integrate jsons by class.
    itg_tag_stats = {}
    for label in user_label:
        itg_tag_stats[label] = {'n_data': 0, 'tags': {}}
        
        for bias in ['align', 'conflict']:
            # Load filtered.json.
            tag_stats_json_path = os.path.join(root_path, 'preprocessed', dataset, conflict_ratio+'pct', bias, label, 'jsons', 'tag_stats.json')
            if os.path.exists(tag_stats_json_path):
                with open(tag_stats_json_path, 'r') as file:
                    try:
                        tag_stats = json.load(file)
                    except json.JSONDecodeError:
                        raise RuntimeError("An error occurred while loading the existing json file.")
            else:
                raise RuntimeError(f"tag_stats.json does not exist.\nPath: {tag_stats_json_path}")
            
            # Combine results from align/conflict.
            itg_tag_stats[label]['n_data'] += tag_stats['n_data']
            for tag in tag_stats['tags']:
                if tag in itg_tag_stats[label]['tags']: 
                    itg_tag_stats[label]['tags'][tag]['appeared'] += tag_stats['tags'][tag][0]
                else:
                    itg_tag_stats[label]['tags'][tag] = {'appeared': tag_stats['tags'][tag][0], 
                                                         'cond1': None, 
                                                         'cond2': None}
                    
            # Bias candidate validation
            # Condition. 1: Should over number of {n_samples_in_label/n_class}
            # Condition. 2: Should number of {n_class/2} classes do not have it.
            # Bias attribute = Condition1 & Condition2
            for tag in itg_tag_stats[label]['tags']:
                condition_1 = 1 if itg_tag_stats[label]['tags'][tag]['appeared'] >= itg_tag_stats[label]['n_data']/len(user_label) else 0
                itg_tag_stats[label]['tags'][tag]['cond1'] = condition_1
        
    for target_label in user_label:
        for tag in itg_tag_stats[target_label]['tags']:
            cond2_cnt = 0
            for subject_label in user_label:
                if tag in itg_tag_stats[subject_label]['tags']:
                    if itg_tag_stats[subject_label]['tags'][tag]['cond1'] == 1:
                        cond2_cnt += 1
            condition_2 = 1 if cond2_cnt <= len(user_label)/2 else 0
            itg_tag_stats[target_label]['tags'][tag]['cond2'] = condition_2

    for label in user_label:
        # Sort tags.  
        itg_tag_stats[label]['tags'] = sort_tags_(itg_tag_stats[label]['tags'])
        bias_candidates = {}
        for tag in itg_tag_stats[label]['tags']:
            itg_tag_stats[label]['tags'][tag]['appeared'] = [itg_tag_stats[label]['tags'][tag]['appeared'], itg_tag_stats[label]['tags'][tag]['appeared']/itg_tag_stats[label]['n_data']]
        for tag in itg_tag_stats[label]['tags']:
            if itg_tag_stats[label]['tags'][tag]['cond1'] and itg_tag_stats[label]['tags'][tag]['cond2']:
                bias_candidates[tag] = itg_tag_stats[label]['tags'][tag]
            if len(bias_candidates) >= n_bias or itg_tag_stats[label]['tags'][tag]['appeared'][-1] <= 1/len(user_label):
                break
        
        itg_tag_stats[label]['bias_candidates'] = bias_candidates
    
    # Save json.
    save_json_path = os.path.join(root_path, 'preprocessed', dataset, conflict_ratio+'pct', 'tag_stats.json')
    with open(save_json_path, 'w') as file:
        json.dump(itg_tag_stats, file, indent=4)

    print("[Done] Bias candidates: tag_stats.json files have been made.")
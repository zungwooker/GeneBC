from tqdm import tqdm
import google.generativeai as genai
import time
import glob
import os
import json
import typing_extensions as typing


class Gemini():
    def __init__(self,
                 args,
                 dataset: str,
                 conflict_ratio: str,
                 class_name: dict[str: str],
                 root_path: str,
                 api_key: str='AIzaSyBnmZlprShV64CTkdWpKIg5U4THzGVS1Xo'):
        genai.configure(api_key=api_key)
        self.args = args
        self.root_path = root_path
        self.api_key = api_key
        self.spare_api_keys = {
            # Most of them had been banned from Google.
            0: 'AIzaSyBnmZlprShV64CTkdWpKIg5U4THzGVS1Xo',
            # 1: 'AIzaSyBXbl51qJctiEgfTsGHIqcCNR7N1WlLtFM',
            # 2: 'AIzaSyDVOQR4HGtQyKrsGTjWaE7ZUSs9ay8fj8Q',
            # 3: 'AIzaSyAYapZ3ooGS6K0Inlcu8eXuhTMp5nj7NGI',
            # 4: 'AIzaSyADgzIO3cyc4JBkWjfGTCG_SjtSjBwoLp0',
            # 5: 'AIzaSyCUMx-AiOVnIleBOXop1h8s75Ae28PoOP8',
            # 6: 'AIzaSyD2JIl3opHLyMsEipUE1_LcRbKhPGRSRy8',
            # 7: 'AIzaSyA8rkcHxlOpgNtnbAruxD2y7BYHpdjW7GY',
            # 8: 'AIzaSyD709-0knP-jK62es0oZn9nE6nPD6Cu8aI',
            # 9: 'AIzaSyDzxpWAtL8nZoDtEriNbMYUiA70QTDT_0k',
            # 10: 'AIzaSyAZwUiNy_YJMYNhL0i4whSpUFh9Maf175c',
            # 11: 'AIzaSyC-21g28gAh0AGntZnT4n79F_Qa2AAJtKI',
            # 12: 'AIzaSyC2tfhW2hQFT9mM77szcz1ge1CDwB6BB1k',
            # 13: 'AIzaSyDkjCQiMQHQLOYWZa2WnE7g-3eACyE91Rs',
            # 14: 'AIzaSyAAiCCOd3z_ex4Qa_zrQsMztOmncSEi87M',
            # 15: 'AIzaSyBSIpNsKpWKt3RmSLVYSMniGoa1s653-94',
            # 16: 'AIzaSyD4Ygq5oVO32KmHBxe4Gycq1YgBFgupI-s',
            # 17: 'AIzaSyDTccsEJivRKANnma4DIVAxgjjVyMtNhhM',
            # 18: 'AIzaSyCor9edv7Pn_QzjifYfnMpvKxqPV8Rv7ak',
            # 19: 'AIzaSyBubEN2Ym4ih-4KudDJRe-yAB58YIOSXKk',
        }
        self.dataset = dataset
        self.conflict_ratio = conflict_ratio
        
        class KeywordPair(typing.TypedDict):
            input_keyword: str
            output_keyword: list[str]
            
        self.gemini = genai.GenerativeModel('gemini-1.5-pro',
                                            generation_config={"response_mime_type": "application/json",
                                                               "response_schema": list[KeywordPair]})
        self.pre_prompt = "\
            ### Instruction ###\
            1. For each given input keyword, generate pairs of keywords with opposite or antonym concepts.\
            2. Ensure that the output keywords belong to the same category as the input keywords.\
            3. [Very Important] Ensure that the input keywords and output keywords are never the same. Double-check to provide accurate answers.\
            4. [Very Important] If an exact opposite concept exists for the input keyword, provide only that single keyword. If no exact opposite concept exists, provide multiple related keywords.\
            5. Using this JSON scheme: KeywordPair = {'input_keyword': str, 'output_keyword': list[str]} and return a list[KeywordPair]\
            \
            ### Correct Example ###\
            Input keywords: turtle, running, night\
            Output:\
            turtle: [rabbit, cat, dog]\
            running: [sitting, eating, talking]\
            long: [short]\
            quite: [noisy]\
            \
            ### Explanation of the example ###\
            Given the input keywords: turtle, running, night. \
            Since 'turtle' and 'running' do not have exact opposite concepts, multiple related keywords are provided.\
            For 'long' and 'quiet', since exact opposite concepts exist, only one keyword is provided.\
            \
            Now, I will provide you with input keywords. \
            Generate appropriate output pairs. \
            This is a very important task, so do not make any mistakes. \
            Double-check your work.\
            \
            ### Input ###\
            "
            
    def generate(self, bias_candidates: list[str]):
        retry_seconds = 10
        retry_counts = 0
        NUM_RETRY = 10
        
        post_prompt = f"Input keywords: {', '.join(bias_candidates)}"
        while retry_counts < NUM_RETRY:
            try:
                response = self.gemini.generate_content(self.pre_prompt + post_prompt)
                break
            except:
                time.sleep(retry_seconds)
                retry_seconds *= 2
        
        if not retry_counts < NUM_RETRY:
            raise Exception("Gemini Error.")
                
        return json.loads(response.text)
    
    def generate_conflict_json(self):
        # Load tag_stats.json.
        tag_stats_json_path = os.path.join(self.root_path, self.args.preproc, self.dataset, self.conflict_ratio+'pct', 'tag_stats.json')
        if os.path.exists(tag_stats_json_path):
            with open(tag_stats_json_path, 'r') as file:
                try:
                    tag_stats = json.load(file)
                except json.JSONDecodeError:
                    raise RuntimeError("An error occurred while loading the existing json file.")
        else:
            raise RuntimeError(f"tag_stats.json does not exist.\nPath: {tag_stats_json_path}")
        
        # Generate conflict keyword pairs.
        for label in tag_stats:
            print(f"[Gemini] Working on label: {label}.")
            
            
            conflict_pairs = self.generate(bias_candidates=tag_stats[label]['bias_candidates'])
            conflict_pairs = {ele['input_keyword']: ele['output_keyword'] for ele in conflict_pairs}
            
            for bias in ['align', 'conflict']:
                filtered_json_path = os.path.join(self.root_path, self.args.preproc, self.dataset, self.conflict_ratio+'pct', bias, label, 'jsons', 'filtered.json')
                if os.path.exists(filtered_json_path):
                    with open(filtered_json_path, 'r') as file:
                        try:
                            filtered_json = json.load(file)
                        except json.JSONDecodeError:
                            raise RuntimeError("An error occurred while loading the existing json file.")
                else:
                    raise RuntimeError(f"filtered.json does not exist.\nPath: {filtered_json_path}")
                
                for image_id in filtered_json:
                    filtered_json[image_id]['conflict_pairs'] = {}
                    for bias_tag in conflict_pairs:
                        if bias_tag in filtered_json[image_id]['filtered_tags']:
                            filtered_json[image_id]['conflict_pairs'][bias_tag] = conflict_pairs[bias_tag]
                            
                save_json_path = os.path.join(self.root_path, self.args.preproc, self.dataset, self.conflict_ratio+'pct', bias, label, 'jsons', 'conflict.json')
                with open(save_json_path, 'w') as file:
                    json.dump(filtered_json, file, indent=4)
        
        print("[Done] Gemini: conflict.json files have been made.")
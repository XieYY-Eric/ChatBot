from logger import Logger

import os
import json
import re
import csv

logger = Logger()

def create_json_from_data(dataset_path, json_path):
    with open(dataset_path, "r") as f:
            lines = f.readlines()
            prompts = lines[0::2]
            completions = lines[1::2]
    pairs = []
    for i in range(len(lines)//2):
        
        # ======================================
        # NOTE(Sean) I kept this from one of my stashed changes, idk if it's still necessary...
        if r'\u' in prompts[i]:
            prompts[i].replace(r'\u', "'")
        
        if r'\u' in completions[i]:
            completions[i].replace(r'\u', "'")
        # ======================================
        
        pairs.append({"prompt" : str(prompts[i]), "completion" : str(completions[i])})

    #Write the object to file.
    logger.log_info(f"Now writing data from '{dataset_path}' to json at '{json_path}'")
    with open(json_path,'w') as jsonFile:
        json.dump(pairs, jsonFile)
    return

def create_csv_from_data(dataset_path, csv_path):
    header = ["prompt", "completion"]
    with open(dataset_path, "r") as f:
            lines = f.readlines()
            prompts = lines[0::2]
            completions = lines[1::2]

    pairs = []
    for i in range(len(lines)//2):
        pairs.append([prompts[i], completions[i]])
    
    logger.log_info(f"Now writing data from '{dataset_path}' to csv at '{csv_path}'")
    with open(csv_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for pair in pairs:
            writer.writerow(pair)
    return

def gtp3_model_setup(dataset_filepath: str = "./data/breaking_bad_dataset.txt", 
    csv_filepath: str = "./data/breaking_bad_csv.csv",
    force: bool = False
    ) -> None:

    if os.path.isfile(csv_filepath) and not force:
        logger.log_info(f"dataset '{csv_filepath}' already exists. (use force=True to overwrite existing dataset)")
        return
    
    if not(os.path.isfile(dataset_filepath)):
        logger.log_info(f"dataset '{dataset_filepath}' does not exist. Please create preprocess the data first.")
        return

    create_csv_from_data(dataset_path=dataset_filepath, csv_path=csv_filepath)
    logger.log_info(f"Finished creating the csv ('{csv_filepath}') from the dataset '{dataset_filepath}'")
    return

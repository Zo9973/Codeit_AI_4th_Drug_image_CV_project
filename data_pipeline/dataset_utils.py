import os
import json
import random
from tqdm import tqdm
from collections import defaultdict

def collect_json_files(annotations_path):
    json_files = []
    for root, dirs, files in os.walk(annotations_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def process_json_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def merge_annotations_by_image(annotations_path):
    json_files = collect_json_files(annotations_path)
    image_to_annots = defaultdict(list)
    categories = {}

    for json_file in tqdm(json_files, desc="Parsing JSON"):
        data = process_json_file(json_file)

        if 'images' in data and 'annotations' in data and 'categories' in data:
            image_dict = {img['id']: img for img in data['images']}
            for ann in data['annotations']:
                img_id = ann['image_id']
                image_info = image_dict.get(img_id)
                if image_info:
                    filename = image_info['file_name']
                    image_to_annots[filename].append((ann, image_info))
            for cat in data['categories']:
                categories[cat['id']] = cat['name']

    return image_to_annots, categories

def split_dataset(image_to_annots, train_ratio=0.9):
    keys = list(image_to_annots.keys())
    random.shuffle(keys)
    split_idx = int(len(keys) * train_ratio)
    return keys[:split_idx], keys[split_idx:]

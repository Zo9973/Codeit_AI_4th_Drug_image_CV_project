import os
import shutil
import yaml
from tqdm import tqdm

def coco_to_yolo_bbox(bbox, image_width, image_height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    width = w / image_width
    height = h / image_height
    return [x_center, y_center, width, height]

def process_image(annot_list, all_categories, category_id_map):
    image_info = annot_list[0][1]
    img_width = image_info['width']
    img_height = image_info['height']

    yolo_annots = []
    for ann, _ in annot_list:
        category_id = ann['category_id']
        yolo_id = category_id_map.get(category_id, -1)
        if yolo_id == -1:
            continue
        yolo_bbox = coco_to_yolo_bbox(ann['bbox'], img_width, img_height)
        yolo_annots.append((yolo_id, yolo_bbox))

    return image_info['file_name'], yolo_annots

def convert_to_yolo_format(image_to_annots, all_categories, category_id_map,
                           save_img_dir, save_label_dir, img_root_dir):
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    for filename, annots in tqdm(image_to_annots.items(), desc="Converting to YOLO"):
        file_name, yolo_annots = process_image(annots, all_categories, category_id_map)
        label_path = os.path.join(save_label_dir, os.path.splitext(file_name)[0] + '.txt')

        with open(label_path, 'w') as f:
            for yolo_id, bbox in yolo_annots:
                f.write(f"{yolo_id} " + " ".join(f"{coord:.6f}" for coord in bbox) + "\n")

        src_img_path = os.path.join(img_root_dir, file_name)
        dst_img_path = os.path.join(save_img_dir, file_name)
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)

def save_dataset_yaml(yolo_path, category_id_map):
    names = [None] * len(category_id_map)
    for k, v in category_id_map.items():
        names[v] = str(k)  # category_id -> class name (could improve)

    data = {
        'train': f'{yolo_path}/images/train',
        'val': f'{yolo_path}/images/val',
        'nc': len(names),
        'names': names,
    }

    yaml_path = os.path.join(yolo_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

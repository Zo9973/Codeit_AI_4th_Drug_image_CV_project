import torch
from pathlib import Path

# 경로 설정
project_path = Path("C:/Project")
train_img_path = f"{project_path}/data/train_images"
train_ann_path = f"{project_path}/data/train_annotations"
test_img_path = f"{project_path}/data/test_images"
yolo_path = f"{project_path}/data/yolo_dataset_final"

# 결과 저장 경로
result_dir = f"{project_path}/result/train"
csv_output_path = f"{project_path}/result/infer"

# YOLO 학습 설정
training_config = {
    'data': f"{yolo_path}/dataset.yaml",
    'epochs': 100,
    'imgsz': 1280,
    'batch': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'project': result_dir,
    'name': 'pill_detection',
    'save': True,
    'plots': True,
    'verbose': True,
    'augment': True,
}

# YOLO 추론 설정
inference_config = {
    'source': test_img_path,
    'imgsz': 1280,
    'conf': 0.25,
    'iou': 0.45,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    "classes": None,
    "agnostic_nms": False,
    "augment": False,
    "half": False,
    "visualize": False,
    "stream": True,
    'save' : False,
}
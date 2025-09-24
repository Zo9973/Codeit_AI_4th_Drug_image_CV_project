import os
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
from tqdm import tqdm

def get_unique_filename(prefix="inference", ext=".csv"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{ext}"

def run_inference_and_save_csv(best_model_path, inference_config, category_id_map, output_path):
    model = YOLO(best_model_path)
    class_map = {v: k for k, v in category_id_map.items()}

    infer_results = model.predict(**inference_config)

    output_data = []
    for infer_result in tqdm(infer_results, desc="Running inference"):
        img_name = os.path.basename(infer_result.path)
        for box in infer_result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            class_id = int(cls)
            category_id = class_map.get(class_id, -1)
            output_data.append({
                'image_id': img_name,
                'category_id': category_id,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'confidence': conf
            })

    df = pd.DataFrame(output_data)
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, get_unique_filename())
    df.to_csv(csv_path, index=False)
    print(f"✅ 추론 결과 저장됨: {csv_path}")

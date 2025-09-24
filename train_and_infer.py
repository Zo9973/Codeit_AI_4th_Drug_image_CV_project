from src.config import (
    train_img_path, train_ann_path, yolo_path,
    training_config, inference_config, csv_output_path
)

from ultralytics import YOLO
from data_pipeline.dataset_utils import merge_annotations_by_image, split_dataset
from data_pipeline.yolo_utils import convert_to_yolo_format, save_dataset_yaml
from data_pipeline.train_utils import analyze_latest_yolo_training_result
from data_pipeline.inference_utils import run_inference_and_save_csv

import os

def main():  
    # 1. 어노테이션 병합
    image_to_annots, all_categories = merge_annotations_by_image(train_ann_path)

    # 2. 카테고리 매핑
    category_id_map = {cid: idx for idx, cid in enumerate(sorted(all_categories.keys()))}

    # 3. 데이터셋 분할
    train_keys, val_keys = split_dataset(image_to_annots)

    # 4. YOLO 포맷 변환
    convert_to_yolo_format(
        {k: image_to_annots[k] for k in train_keys},
        all_categories,
        category_id_map,
        os.path.join(yolo_path, "images/train"),
        os.path.join(yolo_path, "labels/train"),
        train_img_path
    )

    convert_to_yolo_format(
        {k: image_to_annots[k] for k in val_keys},
        all_categories,
        category_id_map,
        os.path.join(yolo_path, "images/val"),
        os.path.join(yolo_path, "labels/val"),
        train_img_path
    )

    # 5. dataset.yaml 저장
    save_dataset_yaml(yolo_path, category_id_map)

    # 6. YOLO 학습
    model = YOLO("yolov8n.pt")
    try:
        train_results = model.train(**training_config)
        print("✅ 학습 완료!")
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. 학습 결과 분석
    analysis_result = analyze_latest_yolo_training_result()
    best_pt = analysis_result.get('best_model_path')
    if best_pt:
        print(f"\n📌 사용할 모델: {best_pt}")

        # 8. 테스트 이미지 추론
        run_inference_and_save_csv(
            best_model_path=best_pt,
            inference_config=inference_config,
            category_id_map=category_id_map,
            output_path=csv_output_path,
        )
    else:
        print("❌ best.pt 파일을 찾을 수 없습니다.")


if __name__ == "__main__":
    main()

import os
import json
import glob
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
from ultralytics import RTDETR
import yaml
import shutil
from collections import defaultdict
from datetime import datetime
import copy

"""## 전처리 클래스"""

class RTDETRDataProcessor:
    def __init__(self, data_path="./data"):
        self.data_path = data_path
        self.train_images_path = os.path.join(data_path, "train_images")
        self.train_ann_path = os.path.join(data_path, "train_annotations")
        self.test_images_path = os.path.join(data_path, "test_images")

        # YOLO 형식 출력 디렉토리
        self.output_dir = os.path.join(data_path, "yolo_format")
        os.makedirs(self.output_dir, exist_ok=True)

    def merge_coco_annotations(self):
        """COCO 형식 JSON 파일들을 하나로 병합 + 모든 dl_idx 매핑 저장 (수정된 버전)"""
        pattern = os.path.join(self.train_ann_path, "**", "*.json")
        json_files = glob.glob(pattern, recursive=True)

        merged_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # ID 재할당을 위한 매핑
        image_id_map = {}
        category_name_to_id = {}
        new_image_id = 0
        new_ann_id = 0
        new_cat_id = 0

        # 🔥 핵심: 모든 매핑을 저장하는 구조
        dl_idx_to_name_mapping = {}    # 모든 dl_idx → 약품명
        category_mapping = {}          # YOLO 클래스 ID → dl_idx
        processed_dl_idx = set()       # 중복 방지

        print(f"총 {len(json_files)}개의 JSON 파일 처리 중...")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 파일 형식 확인
                categories = data.get('categories', [])
                is_broken_format = (len(categories) == 1 and
                                categories[0].get('name') == 'Drug' and
                                categories[0].get('id') == 1)

                if is_broken_format:
                    print(f"  새로운 형식: {os.path.basename(json_file)}")

                    # 새로운 형식: 이미지에서 dl_idx와 dl_name 추출
                    for img in data.get('images', []):
                        actual_dl_idx = img.get('dl_idx')
                        actual_dl_name = img.get('dl_name')

                        if actual_dl_idx and actual_dl_name:
                            if isinstance(actual_dl_idx, str):
                                actual_dl_idx = int(actual_dl_idx)

                            if actual_dl_idx not in processed_dl_idx:
                                processed_dl_idx.add(actual_dl_idx)

                                # 🔥 수정: 모든 매핑 저장
                                dl_idx_to_name_mapping[actual_dl_idx] = actual_dl_name

                                if actual_dl_name not in category_name_to_id:
                                    category_name_to_id[actual_dl_name] = new_cat_id
                                    category_mapping[str(new_cat_id)] = actual_dl_idx

                                    merged_data['categories'].append({
                                        'id': new_cat_id,
                                        'name': actual_dl_name,
                                        'supercategory': 'pill',
                                        'original_dl_idx': actual_dl_idx
                                    })

                                    print(f"    + {actual_dl_name} (dl_idx: {actual_dl_idx})")
                                    new_cat_id += 1

                else:
                    print(f"  기존 형식: {os.path.basename(json_file)}")

                    # 🔥 수정: 기존 형식도 매핑 저장
                    for cat in categories:
                        cat_name = cat['name']
                        original_dl_idx = cat['id']

                        if original_dl_idx not in processed_dl_idx:
                            processed_dl_idx.add(original_dl_idx)

                            # 🔥 핵심: 기존 데이터도 매핑 저장
                            dl_idx_to_name_mapping[original_dl_idx] = cat_name

                            if cat_name not in category_name_to_id:
                                category_name_to_id[cat_name] = new_cat_id
                                category_mapping[str(new_cat_id)] = original_dl_idx

                                merged_data['categories'].append({
                                    'id': new_cat_id,
                                    'name': cat_name,
                                    'supercategory': 'pill',
                                    'original_dl_idx': original_dl_idx
                                })

                                print(f"    + {cat_name} (dl_idx: {original_dl_idx})")
                                new_cat_id += 1

                # 이미지 처리 (기존과 동일)
                for img in data.get('images', []):
                    old_img_id = img['id']
                    image_id_map[old_img_id] = new_image_id
                    img['id'] = new_image_id
                    merged_data['images'].append(img)
                    new_image_id += 1

                # 어노테이션 처리 (수정됨)
                for ann in data.get('annotations', []):
                    ann['id'] = new_ann_id
                    ann['image_id'] = image_id_map[ann['image_id']]

                    if is_broken_format:
                        # 새로운 형식: 이미지에서 dl_idx 찾기
                        original_img_id = None
                        for orig_id, mapped_id in image_id_map.items():
                            if mapped_id == ann['image_id']:
                                original_img_id = orig_id
                                break

                        target_img = next((img for img in data['images'] if img['id'] == original_img_id), None)
                        if target_img and target_img.get('dl_idx'):
                            actual_dl_idx = int(target_img['dl_idx']) if isinstance(target_img['dl_idx'], str) else target_img['dl_idx']
                            actual_dl_name = target_img.get('dl_name')

                            if actual_dl_name in category_name_to_id:
                                ann['category_id'] = category_name_to_id[actual_dl_name]
                                merged_data['annotations'].append(ann)
                                new_ann_id += 1
                    else:
                        # 기존 형식
                        old_cat_id = ann['category_id']
                        target_cat = next((cat for cat in data['categories'] if cat['id'] == old_cat_id), None)

                        if target_cat and target_cat['name'] in category_name_to_id:
                            ann['category_id'] = category_name_to_id[target_cat['name']]
                            merged_data['annotations'].append(ann)
                            new_ann_id += 1

            except Exception as e:
                print(f"❌ 에러: {json_file} - {e}")
                continue

        # 🔥 수정: 모든 매핑 파일 저장
        print(f"\n📝 매핑 파일 저장 중...")

        # category_id_mapping.json (YOLO 클래스 → dl_idx)
        with open(os.path.join(self.output_dir, 'category_id_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump(category_mapping, f, ensure_ascii=False, indent=2)

        # dl_idx_to_name.json (dl_idx → 약품명)
        with open(os.path.join(self.output_dir, 'dl_idx_to_name.json'), 'w', encoding='utf-8') as f:
            json.dump(dl_idx_to_name_mapping, f, ensure_ascii=False, indent=2)

        # dl_name_mapping.json (동일한 내용이지만 호환성을 위해)
        with open(os.path.join(self.output_dir, 'dl_name_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump(dl_idx_to_name_mapping, f, ensure_ascii=False, indent=2)

        print(f"\n🎯 병합 결과:")
        print(f"  📊 총 카테고리: {len(merged_data['categories'])}개")
        print(f"  📊 총 이미지: {len(merged_data['images'])}개")
        print(f"  📊 총 어노테이션: {len(merged_data['annotations'])}개")
        print(f"  📊 매핑된 dl_idx: {len(dl_idx_to_name_mapping)}개")

        # 매핑 검증
        if len(dl_idx_to_name_mapping) != len(merged_data['categories']):
            print(f"  ⚠️  매핑 불일치 감지!")
            print(f"     카테고리: {len(merged_data['categories'])}개")
            print(f"     매핑: {len(dl_idx_to_name_mapping)}개")
        else:
            print(f"  ✅ 매핑 일치 확인")

        return merged_data

    def _find_actual_drug_name(self, dl_idx, data, dl_name_map):
        """dl_idx를 기반으로 실제 약품명 찾기"""

        # 방법 1: 해당 dl_idx를 가진 이미지들의 dl_name 확인
        for img in data.get('images', []):
            if img.get('dl_name') and 'annotations' in data:
                # 이 이미지의 어노테이션 중 해당 dl_idx를 가진 것 찾기
                for ann in data['annotations']:
                    if ann['image_id'] == img['id'] and ann['category_id'] == dl_idx:
                        return img.get('dl_name')

        # 방법 2: dl_name_map에서 직접 찾기
        if dl_name_map:
            for filename, drug_name in dl_name_map.items():
                # 파일명 패턴으로 매칭 시도
                if str(dl_idx) in filename:
                    return drug_name

        return None  # 찾지 못한 경우

    def validate_category_mapping(self):
        """카테고리 매핑 검증"""
        mapping_file = os.path.join(self.output_dir, 'category_id_mapping.json')
        dl_name_file = os.path.join(self.output_dir, 'dl_name_mapping.json')

        if os.path.exists(mapping_file) and os.path.exists(dl_name_file):
            with open(mapping_file, 'r') as f:
                category_mapping = json.load(f)
            with open(dl_name_file, 'r') as f:
                dl_name_mapping = json.load(f)

            print(f"\n🔍 매핑 검증:")
            print(f"  - 총 카테고리: {len(category_mapping)}개")
            print(f"  - dl_name 매핑: {len(dl_name_mapping)}개")

            # "Drug"가 남아있는지 확인
            generic_drugs = [name for name in dl_name_mapping.values() if "Drug_idx_" in name]
            if generic_drugs:
                print(f"  ⚠️  백업명 사용: {len(generic_drugs)}개")
                for drug in generic_drugs[:3]:  # 처음 3개만 출력
                    print(f"      {drug}")
            else:
                print(f"  ✅ 모든 약품명 매핑 완료")

        return True

    def coco_to_yolo_format(self, merged_data):
        """COCO 형식을 YOLO 형식으로 변환 + 매핑 파일 생성"""
        # 디렉토리 생성
        train_dir = os.path.join(self.output_dir, "train")
        val_dir = os.path.join(self.output_dir, "val")
        os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)

        # 이미지 분할 (80:20)
        images = merged_data['images']
        np.random.seed(42)
        np.random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # 어노테이션을 이미지별로 그룹화
        ann_by_image = {}
        for ann in merged_data['annotations']:
            img_id = ann['image_id']
            if img_id not in ann_by_image:
                ann_by_image[img_id] = []
            ann_by_image[img_id].append(ann)

        # 훈련 데이터 처리
        self._process_split(train_images, ann_by_image, train_dir, "train")

        # 검증 데이터 처리
        self._process_split(val_images, ann_by_image, val_dir, "val")

        # 🔥 핵심: category_id 매핑 파일 생성 (제출용)
        category_mapping = {}
        original_dl_idx_mapping = {}

        for i, cat in enumerate(merged_data['categories']):
            # YOLO 훈련용 ID (0,1,2,3...)
            yolo_class_id = i

            # 실제 dl_idx 추출 (원본 JSON에서 보존된 값)
            original_dl_idx = cat.get('original_dl_idx', cat['id'])

            category_mapping[str(yolo_class_id)] = original_dl_idx
            original_dl_idx_mapping[original_dl_idx] = cat['name']

        # 매핑 파일들 저장
        with open(os.path.join(self.output_dir, 'category_id_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump(category_mapping, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.output_dir, 'dl_idx_to_name.json'), 'w', encoding='utf-8') as f:
            json.dump(original_dl_idx_mapping, f, ensure_ascii=False, indent=2)

        print(f"\n🔥 매핑 파일 생성 완료:")
        print(f"  - category_id_mapping.json: YOLO 클래스 → dl_idx 매핑")
        print(f"  - dl_idx_to_name.json: dl_idx → 약품명 매핑")
        print(f"  - 총 {len(category_mapping)}개 클래스")

        # 매핑 예시 출력
        print(f"매핑 예시:")
        for yolo_id, dl_idx in list(category_mapping.items())[:5]:
            drug_name = original_dl_idx_mapping.get(dl_idx, "Unknown")
            print(f"  YOLO 클래스 {yolo_id} → dl_idx {dl_idx} ({drug_name})")

        # dataset.yaml 파일 생성
        yaml_data = {
            'path': self.output_dir,
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(merged_data['categories']),
            'names': [cat['name'] for cat in merged_data['categories']]
        }

        with open(os.path.join(self.output_dir, 'dataset.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True)

        return merged_data

    def _process_split(self, images, ann_by_image, output_dir, split_name):
        """분할된 데이터 처리 (bbox 유효성 검사 포함)"""
        skipped_annotations = 0
        processed_annotations = 0

        for img_data in images:
            img_id = img_data['id']
            filename = img_data['file_name']

            # 이미지 복사
            src_path = os.path.join(self.train_images_path, filename)
            dst_path = os.path.join(output_dir, "images", filename)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

            # YOLO 라벨 생성
            label_filename = filename.replace('.png', '.txt').replace('.jpg', '.txt')
            label_path = os.path.join(output_dir, "labels", label_filename)

            with open(label_path, 'w') as f:
                if img_id in ann_by_image:
                    for ann in ann_by_image[img_id]:
                        # 🔥 bbox 유효성 검사 추가
                        bbox = ann.get('bbox', [])
                        if not bbox or len(bbox) != 4:
                            print(f"⚠️ 잘못된 bbox 건너뛰기: {bbox} (이미지: {filename})")
                            skipped_annotations += 1
                            continue

                        # COCO bbox를 YOLO 형식으로 변환
                        x, y, w, h = bbox

                        # 🔥 추가 유효성 검사 (음수 크기 방지)
                        if w <= 0 or h <= 0:
                            print(f"⚠️ 유효하지 않은 크기 건너뛰기: w={w}, h={h} (이미지: {filename})")
                            skipped_annotations += 1
                            continue

                        # 이미지 크기 정보
                        img_w, img_h = img_data['width'], img_data['height']

                        # 🔥 이미지 경계 체크
                        if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
                            print(f"⚠️ 이미지 경계 초과 bbox 건너뛰기: ({x}, {y}, {w}, {h}) (이미지: {filename})")
                            skipped_annotations += 1
                            continue

                        # 정규화된 중심점 좌표와 크기
                        x_center = (x + w/2) / img_w
                        y_center = (y + h/2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h

                        # 🔥 정규화된 값 범위 체크 (0~1 사이)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= norm_w <= 1 and 0 <= norm_h <= 1):
                            print(f"⚠️ 정규화 값 범위 초과: center=({x_center:.3f}, {y_center:.3f}), size=({norm_w:.3f}, {norm_h:.3f})")
                            skipped_annotations += 1
                            continue

                        class_id = ann['category_id']
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                        processed_annotations += 1

        # 🔥 처리 결과 요약
        print(f"\n📊 {split_name} 데이터 처리 완료:")
        print(f"  - 처리된 어노테이션: {processed_annotations}개")
        print(f"  - 건너뛴 어노테이션: {skipped_annotations}개")
        if skipped_annotations > 0:
            print(f"  - 성공률: {processed_annotations/(processed_annotations + skipped_annotations)*100:.1f}%")

    def process_data_with_name_mapping(self):
        """dl_name 매핑 포함 전체 데이터 처리"""
        print("=== dl_name 매핑 포함 데이터 전처리 시작 ===")

        print("1. COCO 어노테이션 병합 + dl_name 매핑 중...")
        merged_data = self.merge_coco_annotations()  # 개선된 버전

        print("2. 매핑 검증 중...")
        self.validate_category_mapping()

        print("3. YOLO 형식으로 변환 중...")
        self.coco_to_yolo_format(merged_data)

        print("✅ 데이터 전처리 완료 (dl_name 매핑 포함)!")
        return merged_data

    def quick_mapping_check(self):
        """빠른 매핑 상태 확인"""
        mapping_files = [
            'category_id_mapping.json',
            'dl_name_mapping.json',
            'dl_idx_to_name.json'
        ]

        print("🔍 매핑 파일 상태:")
        all_exist = True
        for filename in mapping_files:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"  ✅ {filename}: {len(data)}개 항목")
            else:
                print(f"  ❌ {filename}: 없음")
                all_exist = False

        return all_exist

    def process_data(self):
        """전체 데이터 처리 파이프라인 + 클래스 불균형 해결"""
        print("COCO 어노테이션 병합 중...")
        merged_data = self.merge_coco_annotations()

        print("YOLO 형식으로 변환 중...")
        self.coco_to_yolo_format(merged_data)
        print("데이터 전처리 완료!")

        return merged_data

    def analyze_class_distribution(self, merged_data):
        """클래스별 데이터 분포 분석"""
        print("=== 클래스 분포 분석 ===")

        # 클래스별 이미지 수 계산
        class_image_count = defaultdict(int)
        class_annotation_count = defaultdict(int)

        # 이미지별 클래스 집계
        ann_by_image = {}
        for ann in merged_data['annotations']:
            img_id = ann['image_id']
            if img_id not in ann_by_image:
                ann_by_image[img_id] = set()
            ann_by_image[img_id].add(ann['category_id'])
            class_annotation_count[ann['category_id']] += 1

        # 각 클래스가 포함된 이미지 수
        for img_id, classes in ann_by_image.items():
            for class_id in classes:
                class_image_count[class_id] += 1

        # 클래스 이름 매핑
        class_names = {cat['id']: cat['name'] for cat in merged_data['categories']}

        # 분포 출력
        print(f"총 {len(merged_data['categories'])}개 클래스")
        print("\n클래스별 분포 (어노테이션 수 기준):")

        sorted_classes = sorted(class_annotation_count.items(), key=lambda x: x[1], reverse=True)

        minority_classes = []
        majority_classes = []

        for class_id, ann_count in sorted_classes:
            img_count = class_image_count[class_id]
            class_name = class_names.get(class_id, f"Unknown-{class_id}")

            print(f"  {class_name}: {ann_count}개 어노테이션, {img_count}개 이미지")

            if ann_count < 30:  # 30개 미만을 소수 클래스로 정의
                minority_classes.append((class_id, class_name, ann_count))
            elif ann_count > 100:  # 100개 초과를 다수 클래스로 정의
                majority_classes.append((class_id, class_name, ann_count))

        print(f"\n불균형 분석:")
        print(f"  소수 클래스 (<30개): {len(minority_classes)}개")
        print(f"  다수 클래스 (>100개): {len(majority_classes)}개")
        print(f"  불균형 비율: {max(class_annotation_count.values()) / min(class_annotation_count.values()):.1f}배")

        return minority_classes, majority_classes, class_annotation_count

"""# 모델 학습"""

class RTDETRTrainer:
    def __init__(self, data_yaml_path):
        self.data_yaml_path = data_yaml_path
        self.model = None

    def train(self, epochs=100, imgsz=640, batch=16):
        """기본 RT-DETR 모델 학습"""
        print("RT-DETR 모델 학습 시작...")
        self.model = RTDETR('rtdetr-l.pt')

        results = self.model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='pill_rtdetr',
            save_period=10,
            patience=15,       # 조기 종료 완화
            cls=1.0,           # 클래스 손실 가중치 증가
            box=7.5,           # 박스 손실 유지
            dfl=1.5,              # Distribution focal loss (기본 1.5)

            # 학습률 조정
            lr0=0.01,          # 기본값
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,

            # 정규화 강화
            dropout=0.1,       # 드롭아웃 추가

            # 밝기 분포 (95-165) 고려한 증강
            hsv_v=0.2,         # 밝기 변화 제한 (기존보다 작게)
            hsv_s=0.3,         # 채도 변화 적절히
            hsv_h=0.01,        # 색조는 최소화

            # RGB 상관관계 고려
            mixup=0.0,         # 색상 혼합 최소화
            mosaic=0.3,        # 모자이크도 줄임
        )

        print("학습 완료!")
        return results

    def load_best_model(self, model_path=None):
        """최고 성능 모델 로드"""
        if model_path is None:
            model_path = 'runs/detect/pill_rtdetr/weights/best.pt'  # Phase 3 경로

        self.model = RTDETR(model_path)
        return self.model

class RTDETRInference:
    def __init__(self, model_path, mapping_file_path=None):
        self.model = RTDETR(model_path)

        # 🔥 매핑 파일 로드 (YOLO 클래스 → dl_idx)
        self.class_to_dl_idx = {}
        if mapping_file_path and os.path.exists(mapping_file_path):
            with open(mapping_file_path, 'r', encoding='utf-8') as f:
                self.class_to_dl_idx = json.load(f)
                # 문자열 키를 정수로 변환
                self.class_to_dl_idx = {int(k): int(v) for k, v in self.class_to_dl_idx.items()}
            print(f"✅ 매핑 파일 로드 완료: {len(self.class_to_dl_idx)}개 클래스")
        else:
            print("⚠️ 매핑 파일이 없습니다. 클래스 ID를 그대로 사용합니다.")

    def predict_and_generate_csv(self, test_images_path, output_csv_path, conf_thresh=0.321):
        """테스트 이미지 예측 및 제출 규칙에 맞는 CSV 파일 생성"""
        results_data = []
        annotation_id = 1  # 1부터 시작

        # 테스트 이미지 파일 목록
        test_files = glob.glob(os.path.join(test_images_path, "*.png")) + \
                    glob.glob(os.path.join(test_images_path, "*.jpg"))

        print(f"테스트 이미지 {len(test_files)}개 처리 중...")

        for img_path in test_files:
            filename = os.path.basename(img_path)

            # 🔥 image_id: 파일명에서 숫자만 추출 (100.png → 100)
            image_id = filename.split('.')[0]  # 확장자 제거
            try:
                image_id = int(image_id)  # 숫자로 변환
            except ValueError:
                # 숫자가 아닌 파일명의 경우 그대로 사용
                image_id = filename.split('.')[0]

            # 예측 수행
            results = self.model(img_path, conf=conf_thresh)

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 바운딩 박스 좌표 (xyxy 형식)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        yolo_class_id = int(box.cls[0].cpu().numpy())  # YOLO 클래스 ID

                        # 🔥 category_id: YOLO 클래스 → dl_idx 변환
                        if yolo_class_id in self.class_to_dl_idx:
                            category_id = self.class_to_dl_idx[yolo_class_id]  # 실제 dl_idx
                        else:
                            category_id = yolo_class_id  # 매핑 실패시 그대로
                            print(f"⚠️ 매핑 없음: YOLO 클래스 {yolo_class_id} → {yolo_class_id}")

                        # COCO 형식으로 변환 (xywh)
                        bbox_x = x1
                        bbox_y = y1
                        bbox_w = x2 - x1
                        bbox_h = y2 - y1

                        results_data.append({
                            'annotation_id': annotation_id,  # 1부터 시작
                            'image_id': image_id,            # 파일명에서 숫자 추출
                            'category_id': category_id,      # 실제 dl_idx
                            'bbox_x': bbox_x,
                            'bbox_y': bbox_y,
                            'bbox_w': bbox_w,
                            'bbox_h': bbox_h,
                            'score': confidence
                        })

                        annotation_id += 1

        # CSV 파일 생성
        df = pd.DataFrame(results_data)
        df.to_csv(output_csv_path, index=False)

        # 🔥 결과 분석 출력
        if len(results_data) > 0:
            unique_image_ids = df['image_id'].unique()
            unique_category_ids = df['category_id'].unique()

            print(f"\n🎯 제출 파일 생성 완료: {output_csv_path}")
            print(f"  - 총 예측: {len(results_data)}개")
            print(f"  - 이미지 수: {len(unique_image_ids)}개")
            print(f"  - 카테고리 수: {len(unique_category_ids)}개")
            print(f"  - image_id 범위: {min(unique_image_ids)} ~ {max(unique_image_ids)}")
            print(f"  - category_id (dl_idx): {sorted(unique_category_ids)}")
        else:
            print("⚠️ 예측 결과가 없습니다. 신뢰도 임계값을 낮춰보세요.")

        return df

"""# 메인 실행"""

def run_detailed_pipeline(data_path="./data"):
    """상세 버전 파이프라인 실행 (클래스 기반)"""
    print("=== 상세 버전 파이프라인 시작 ===")

    # 1. 데이터 전처리
    print("1. 데이터 전처리...")
    # 전처리 실행
    processor = RTDETRDataProcessor(data_path)
    merged_data = processor.process_data_with_name_mapping()

    # 매핑 결과 확인
    processor.validate_category_mapping()

    2. 모델 학습
    print("2. 모델 학습...")
    trainer = RTDETRTrainer(f"{data_path}/yolo_format/dataset.yaml")
    trainer.train(epochs=50, batch=8)

    # 3. 추론 및 제출 파일 생성
    print("3. 추론 및 제출 파일 생성...")
    model_path = 'runs/detect/pill_rtdetr4/weights/best.pt'
    mapping_file = f"{data_path}/yolo_format/category_id_mapping.json"  # 🔥 매핑 파일 경로

    inference = RTDETRInference(model_path, mapping_file)

    submission_df = inference.predict_and_generate_csv(
        test_images_path=f"{data_path}/test_images",
        output_csv_path="submission_detailed_0.321.csv"
    )

    print("상세 버전 파이프라인 완료!")
    return submission_df

data_path="./data"
run_detailed_pipeline(data_path)


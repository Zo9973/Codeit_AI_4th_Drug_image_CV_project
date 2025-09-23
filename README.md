# 📌 [코드잇 스프린트_AI_4기] 초급 팀프로젝트: 경구약제 이미지 객체 인식 모델
---
코드잇 스프린트 초급 팀프로젝트이다. 이번 프로젝트의 목표는 사진 속에 있는 최대 4개의 알약의 이름(클래스)과 위치(바운딩 박스)를 검출하는 것이다. 또한 하이퍼파라미터 튜닝 등을 통해 최고 성능의 모델을 개발하는 것이 목표이다.
팀은 5인 1팀으로 Project Manager / Data Engineer / Model Architect + Experimentation Lead 로 구성되어있다.
**모델 점수와 상관없이 실무의 팀 개발을 체험하는 과정으로 좋은 인맥 형성, 소프트 스킬 향상, 최선의 팀 결과물 완성이 코드잇에서 말하는 목표이다.**

## 프로젝트 기간: 25.09.09 15:00 ~ 25.09.25 23:50

## kaggle 및 데이터셋 링크
[프로젝트 자료 링크](https://www.kaggle.com/competitions/ai04-level1-project/data)

## 개인 역할

|역할|담당자|업무|
|----|-----|-----|
|Project Manager|신승목|프로젝트 일정관리, 진행상태 확인 및 종합. 최종 보고서 작성, 개발 지원(YOLOv8모델 및 성능 개선 실험, EDA)|
|Data Enginner|지동진|데이터 파이프라인 구축, 데이터 EDA 시행, 파이프라인 자동화, github 및 협업환경 구성 및 관리(RT-DETR모델링, 성능개선 시험)|
|Model Architect|이재영|YOLOv11 모델 담당(모델링, 성능 개선 실험진행)|
|Model Architect + Experimentation Sub|남경민|필요시 지원|
|Experimentation Lead|이솔형|YOLOv8 모델 성능 개선 실험 진행|

---

## 팀프로젝트 수칙
### 1. 데이터 사용 규칙
- 제공된 데이터셋 외 외부 데이터 사용 가능
### 2. 모델 및 코드 제출
- 제출 파일 형식을 준수해 주세요.
- 모델 및 결과물의 재현 가능성을 확보해 주세요.
### 3. 평가 기준 및 리더보드 운영
- 평가 지표: mAP(mean Average Precision)
- 1일 최대 제출 횟수: 5회
- 리더보드는 Public / Private Score로 운영합니다 (최종 순위: Private Score 기준)
---

## 📂 폴더 구성
```
Codeit_AI_4th_Drug_image_CV_project/
├── data/                        # 실제 데이터는 GitHub에 포함되지 않으며,
│   └── data.txt                 # Google Drive 내 데이터 공유 링크가 담긴 텍스트 파일만 존재
├──data_pipeline/                # 데이터 전처리 파이프라인__(자동화)__
│   ├── pipeline.py                # 메인 실행 파일
│   ├── data_preprocess/           # 전처리 모듈
│   │   ├── data_main.py           # 전처리 전용 실행기
│   │   └── modules/               # 핵심 처리 모듈들
│   │       ├── validation.py      # 데이터 검증
│   │       ├── preprocessing.py   # 이미지 전처리
│   │       ├── annotation_converter.py  # COCO→YOLO 변환
│   │       ├── dataset_split.py   # 데이터셋 분할
│   │       ├── data_analyzer.py   # 품질 분석
│   │       └── data_packager.py   # 최종 패키징
│   └── pruning_dataset/           # 소규모 데이터셋 생성
│       ├── pruning_main.py        # 데이터 추출 전용 실행기
│       └── modules/               # 데이터 추출 모듈들
│           ├── config.py          # 설정 관리
│           ├── dataset_splitter.py # 데이터 추출 엔진
│           └── run_dataset.py     # CLI 도구
├── notebooks/                   # Jupyter 노트북
│   ├── data_EDA.ipynb           # 데이터 EDA 보고서
│   └── data_pipeline.ipynb      # 데이터 전처리 관련 코드 작성
├── model/                       # 모델 관련 코드
│   ├── notebooks/               # Jupyter 노트북
│   │   ├── ETL_test_YOLOv8.ipynb  # 파이프라인 작동 테스트용 모델 코드
├── result/                      # 모델학습 및 예측 평가 결과물 탑재
├── LICENSE                      # 라이센스
├── README.md                    # 프로젝트 문서
└── git_clone.ipynb              # Git clone 실습 코드 
```

---
## 개인 협업 일지
- [신승목](https://www.notion.so/1-26919fbbffa18051baa4c561e60e4359?source=copy_link)

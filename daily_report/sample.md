# 개인 협업일지
이곳은 개인별 일일 단위 협업일지를 올리는 곳입니다. 간단한 내용이라도 좋으니 이곳에 업무 결과를 업로드 해주시기 바립니다. 
팀장님께서는 조원들이 일일단위 업무가 잘 진행이 되었는지 확인하시어 피드백 주시면 되겠습니다.  
---
## 문법 정리 사이트

https://inpa.tistory.com/entry/MarkDown-%F0%9F%93%9A-%EB%A7%88%ED%81%AC%EB%8B%A4%EC%9A%B4-%EB%AC%B8%EB%B2%95-%F0%9F%92%AF-%EC%A0%95%EB%A6%AC
https://velog.io/@phobos90/%EB%A7%88%ED%81%AC%EB%8B%A4%EC%9A%B4%EB%AC%B8%EB%B2%95%EC%A0%95%EB%A6%AC

__형식과 양식은 자유이니 편하게 작성하시면 되겠습니다.__
---
신승목
### 9/23
- model_notebooks/YOLOv8_submission_SHIN.iynb 파일 업로드 : 파이프라인 파일
- results/신승목_추가EDA/count_max_jsons.ipynb 파일 업로드 : 추가 EDA 파일
- 내용 : def count_max_jsons(data_path)
- 이미지의 naming 규칙을 분석하여 계산한 json 파일의 이상적인 최대값 계산
- 이미 train 이미지에 최소 하나의 json 파일이 매칭되고, json파일만 있거나 이미지 파일만 있지는 않다는 것을 확인한 상태에서 수행한 분석이다.
- 여섯 자리 숫자로 이미지를 구성하는 셋 또는 네 개의 알약의 고유번호가 이미지와 json파일에 naming 규칙에 반영되어 있다는 점을 이용하여 train 이미지의 이름을 분석하여 각각의 이미지 파일에 연결될 수 있는 이상적인 annotation 수를 계산한다.
- 그리고 실제 데이터셋에 제공된 json 파일 수와 비교하여 데이터셋이 얼마나 충분한지 확인할 수 있다.
- json 파일 계산할 때는 속도를 위해 병렬 계산 방식인 ThreadPoolExecutor를 이용하였고 진행 속도 파악을 위해 tqdm을 적용하였다. (from concurrent.futures import ThreadPoolExecutor, as_completed 필요)
- 계산 결과 : 이미지에 연결될 수 있는 json 파일 최대 개수는 5662개인데 제공된 json 파일 수는 4526개로 79.94% 제공되었다.
- 데이터셋에서 서너개의 약의 조합에 대해 70, 75, 90 형태로 세 번 이미지가 존재하기 때문에 이 정도 json 파일이 제공된다면 클래스 불균형은 존재하더라도 학습 수행에는 충분히 데이터가 제공되었다고 생각한다.

- 
---
이솔형
- 9/17 Run for metrics / YOLOv8s (epoch 300, imgsz 1280, batch 12, optimizer SGD & AdamW, lr 0.01 n 0.001)
---
남경민
- https://www.notion.so/26b492a41e1380698408ce2e98850420?source=copy_link

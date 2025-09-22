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
- 9/16 YOLOv8s (epoch 100, imgsz 1280, batch 8, 전처리 YOLO 기본값), 공유 경로 : results/train/pill_detection_shin
- 9/17 17시 작업 중인 내용 : YOLO 학습 결과로 나온 weights/best.pt로 테스트 이미지를 처리했을 때 Kaggle에 제출할 수 있는 형태로 만들 수 있는 코드 짜는 중
- 9/22 9시 다른 파이프라인으로 데이터셋 만들어서 학습하고 COCO 학습 시 YOLO 인덱싱 변환과 역변환 정보를 저장하여 테스트 이미지로 추론 시 얻은 클래스 정보를 역변환해서 COCO의 정보로 바꿔서 했을 때 점수 나오는 것 확인, 1450개 검출해서 0.44512(epoch 100) 얻었고, clahe+desharp 적용한 이미지(시각적으로 음각이 뚜렷해진 것은 확인)로 추가 데이터셋 만들어서 실행했을 때 검출 개수 984개로 오히려 감소하고 점수 0.27565(epoch 100)
---
이솔형
- 9/17 Run for metrics / YOLOv8s (epoch 300, imgsz 1280, batch 12, optimizer SGD & AdamW, lr 0.01 n 0.001)

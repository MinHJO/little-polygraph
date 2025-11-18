# little-polygraph

## 1. 프로젝트 소개
   
   사람들이 거짓말을 할 때 표정 변화와 목소리의 변화를 분석해서 주어진 영상에서의 인물이 거짓말을 하고 있는지 판단하려고 한다.
   
# 2. 데이터 및 구조
   
   <img width="950" height="450" alt="image" src="https://github.com/user-attachments/assets/25ef7a00-9222-4746-ab7f-1864922530ba" />
   
   Truth : 60    Lie : 61
   
   출처: Mohamed Abouelenien, Associate Professor
Computer & Information Science University of Michigan – Dearborn, https://public.websites.umich.edu/~zmohamed/resources.html

   <img width="1425" height="766" alt="image" src="https://github.com/user-attachments/assets/b6ff1dc4-68a3-4065-a1ab-d719f9761d31" />
   
   Truth: 8      Lie : 17
   
   출처: Kaggle, https://www.kaggle.com/datasets/tamairamirezgordillo/dataset

   <img width="1778" height="965" alt="image" src="https://github.com/user-attachments/assets/1856c4ea-13be-420a-bd66-1981a29f59cf" />

   Truth : 34    Lie : 34      (Only Frames)
   
   출처: Kaggle, https://www.kaggle.com/datasets/devvratmathur/micro-expression-dataset-for-lie-detection

   Total

   Truth : 102   Lie : 112


   과정
   <img width="1344" height="598" alt="image" src="https://github.com/user-attachments/assets/3c0c7b8f-d1d9-4e99-8a4a-bbbe79b3e257" />


   특징 추출
   
   Face Features
   
   <img width="1000" height="445" alt="image" src="https://github.com/user-attachments/assets/ee6a6c5a-7440-4ad9-867c-5953a462ce1a" />

거짓말을 할 때 얼굴에서는 눈썹, 입술, 표정 찡그림 등의 미세한 표현들이 발견된다. 이를 이용하여 영상에서 프레임별로 끊어 변화를 관측한다.
   
   Audio Features
   
   <img width="935" height="532" alt="image" src="https://github.com/user-attachments/assets/0cc4cabc-1d48-47bb-afb7-616a43bdc6af" />

마찬가지로 음성에서의 목소리의 떨림, 높낮이의 변화, MFCC의 변화를 관측한다.

   Face Features 평균 추출값
   <img width="920" height="442" alt="image" src="https://github.com/user-attachments/assets/c33ab241-0e4c-4643-ac43-0a5daefee85a" />

   Audio Features 평균 추출값
   <img width="1117" height="501" alt="image" src="https://github.com/user-attachments/assets/d4f7e7fc-1a5d-414e-8f77-9229489f33b7" />

   Micro Expression 분류 기준
   
   |Frown|          Eyebrows Frown < 15|
   |--------|------------------------|
   |Eyebrows Raise|     Eyebrows Raise  > 20|
   |Lips Up|                  Lips up < 10|
   |Lips Protruded|       Lips Protruded > 5|
   |Head Turn|             Head Turn > 10|


# 3. 모델 평가

   프레임별 변화와 음성의 변화를 csv 파일로 저장하여 모델의 평가를 진행하였다.

   3-1. CNN
   
   |accuracy|val_acc|loss|val_loss|
   |------|------|------|------|
   |0.961|0.9286|0.1759|0.178|

   3-2. GRU
   
   |accuracy|val_acc|loss|val_loss|
   |------|------|------|------|
   |0.934|0.8214|0.2167|0.3149|

   3.3. NetVLAD

   |accuracy|val_acc|loss|val_loss|
   |------|------|------|------|
   |0.9455|0.8929|0.1833|0.1821|

   3-4. BiGRU + Δxₜ

   |accuracy|val_acc|loss|val_loss|
   |------|------|------|------|
   |0.9909|0.9286|0.0939|0.2381|

   3-5. NetVLAD + Δxₜ

   |accuracy|val_acc|loss|val_loss|
   |------|------|------|------|
   |0.9545|0.9286|0.1890|0.1392|


   프레임별 차이를 위해서 Δxₜ를 사용했으며, 같은 데이터를 사용하여 여러 모델을 학습시켰다.
   학습에 필요한 데이터가 적었기에 적은 데이터로 학습이 가능한 NetVLAD를 사용해봤는데 예상보다 좋은 결과를 나타낸다. NetVLAD가 순간적인 패턴 분포를 분석하기에 좋은 결과를 가져온 것으로 예상한다.

   3-6. 모델 분석 요약
   
<img width="870" height="234" alt="image" src="https://github.com/user-attachments/assets/21052891-c69d-45f5-bddb-029d81940142" />

# 4. Test

테스트는 예상외로 좋은 결과를 보여준 NetVLAD 모델을 이용하여 진행하였다
    
테스트에 사용한 영상은 드라마나 예능 속의 장면을 이용하였다.

<img width="465" height="365" alt="image" src="https://github.com/user-attachments/assets/275dc13d-0115-491f-912e-6f2055ee87a6" />

  아리아나 그란데 인터뷰 (진실)
      
      
  |예측 결과| 진실|
  |-----|------|
  |거짓말일 확률| 0.0202|  
  |진실일 확률| 0.9798|

<img width="1399" height="674" alt="image" src="https://github.com/user-attachments/assets/9a031bd8-2942-44f6-b4b1-53f3b59eb778" />

  무한도전 박명수 거짓말 탐지기 (거짓말)

  |예측 결과| 거짓말|
  |-----|------|
  |거짓말일 확률| 0.9845| 
  |진실일 확률| 0.0155|

<img width="650" height="450" alt="image" src="https://github.com/user-attachments/assets/e0a4122c-cc36-46f7-abf1-8be47a333cd6" />

  타짜 고니 마지막 장면 (거짓말)
      
  |예측 결과| 진실|
  |-----|------|
  |거짓말일 확률| 0.0284|
  |진실일 확률| 0.9716|

   드라마의 장면을 테스트 했을 때는 결과가 잘 못 나오는 경우가 더러 있었다. 배우의 연기 실력과 드라마나 영화의 특성 상 배경 음악과 잦은 장면 변화로 인해 테스트가 어려운 것으로 보인다.

# 5. 결론

거짓말 탐지를 위해서 뇌파, 심전도, 영상 및 음성 등 여러가지 모드를 합성하여 사용하고 있다. 

# 참고문헌

[Deception Detection in Videos], Zhe Wu, Bharat Singh, Larry S. Davis, V. S. Subrahmanian, University of Maryland Dartmouth College, 2018


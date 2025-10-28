# little-polygraph

1. 프로젝트 소개
   
   사람들이 거짓말을 할 때 표정 변화와 목소리의 변화를 분석해서 주어진 영상에서의 인물이 거짓말을 하고 있는지 판단하려고 한다.
   
3. 데이터 및 구조
   <img width="1279" height="777" alt="image" src="https://github.com/user-attachments/assets/25ef7a00-9222-4746-ab7f-1864922530ba" />
   
   Truth : 60    Lie : 61
   
   출처: Mohamed Abouelenien, Associate Professor
Computer & Information Science University of Michigan – Dearborn, https://public.websites.umich.edu/~zmohamed/resources.html

   <img width="1425" height="766" alt="image" src="https://github.com/user-attachments/assets/b6ff1dc4-68a3-4065-a1ab-d719f9761d31" />
   
   Truth: 8      Lie : 17
   
   출처: Kaggle, https://www.kaggle.com/datasets/tamairamirezgordillo/dataset

   <img width="1778" height="965" alt="image" src="https://github.com/user-attachments/assets/1856c4ea-13be-420a-bd66-1981a29f59cf" />

   Turht : 34    Lie : 34
   
   출처: Kaggle, https://www.kaggle.com/datasets/devvratmathur/micro-expression-dataset-for-lie-detection


   과정
   <img width="1344" height="598" alt="image" src="https://github.com/user-attachments/assets/3c0c7b8f-d1d9-4e99-8a4a-bbbe79b3e257" />


   특징 추출
   
   Face Features
   
   <img width="1000" height="445" alt="image" src="https://github.com/user-attachments/assets/ee6a6c5a-7440-4ad9-867c-5953a462ce1a" />

   
   Audio Features
   
   <img width="935" height="532" alt="image" src="https://github.com/user-attachments/assets/0cc4cabc-1d48-47bb-afb7-616a43bdc6af" />


   Face Features 평균 추출값
   <img width="920" height="442" alt="image" src="https://github.com/user-attachments/assets/c33ab241-0e4c-4643-ac43-0a5daefee85a" />

   Audio Features 평균 추출값
   <img width="1117" height="501" alt="image" src="https://github.com/user-attachments/assets/d4f7e7fc-1a5d-414e-8f77-9229489f33b7" />

   Micro Expression
   Frown:                   Eyebrows Frown < 15

   Eyebrows Raise:     Eyebrows Raise  > 20

   Lips Up:                  Lips up < 10

   Lips Protruded:       Lips Protruded > 5

   Head Turn:             Head Turn > 10













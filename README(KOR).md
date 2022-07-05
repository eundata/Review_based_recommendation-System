# 리뷰기반 추천시스템</br>(엔코어 최종 프로젝트 22.04.06 ~ 22.04.29)

## 개요
우리는 아마존에서 판매된 책들의 리뷰데이터를 바탕으로 추천 시스템을 만들고자 했다. 추천 시스템에 리뷰 점수를 추가한 이유는 리뷰 점수의 융합이 정확도를 높인다는 여러 <a href="https://github.com/eundata/Recommendation-System/blob/main/papaers.md">논문</a>의 증거를 기반으로 했습니다. 
사용자 활동으로 인해 도서에 대한 평가 기록이나 리뷰 데이터가 누적되고나면, 매일 밤 모델을 재학습한 후 새로운 모델을 제공합니다. 이렇게 MLOPS를 구현해 보았습니다.
저는 이 프로젝트에서 추천 모델을 구축하는 일을 담당했습니다. (모든 과정은 aws에서 진행되었습니다) 
이 문서에서는 제가 수행한 부분만 설명이 있습니다.  

(아래그림은 데이터 흐름도와 데이터 구성요소입니다.)  
![화면 캡처 2022-05-25 140542](https://user-images.githubusercontent.com/96279383/170183938-9f9af045-8b36-4eec-9ce0-b9de168f2780.png)
![화면 캡처 2022-05-27 175313](https://user-images.githubusercontent.com/96279383/170814511-f4d17dda-1c29-4540-a666-293da8e98168.png)


## 방법
- Review data is applied to the recommendation system after sentiment analysis using the BERT model and scoring.  
- The general recommendation is to use the Alternating Least Square (ALS) Matrix Factorization in collaborative filtering.  

First, we use the <a href = "https://github.com/eundata/Recommendation-System/blob/main/BERT.py">BERT</a> to give a score for the review. Scores range from 1 to 5 points. Then, after averaging the user-given score and review score, <a href = "https://colab.research.google.com/drive/1c61kuUElz8g9N0YpC10Zp0OjYuQJqzHy?usp=sharing"> the final score</a> is made. After that, the model is trained with the final score to make the ALS recommendation model.

<details>
  <summary><b>Why did you have to import the model and csv file from the final score file?</b></summary>

It was built for version control of the model and data according to it.
</details>

Second, through membership registration, we obtained information of which books that users are interested in. In the membership registration, users are given a list of 30 books with many reviews and are asked to choose about 3 of their favorite books. Then, using the <a href='https://github.com/eundata/Recommendation-System/blob/main/Cosine_Similarity.py'>cosine similarity</a>, a book with a high similarity to the book selected by the user is recommended.  
<details>
  <summary><b>Why use cosine similarity?</b></summary>

We plan to use the ALS recommendation model for UBCF. However, the cosine similarity method was first adopted because the information on the registered data is not included in the current model and it takes too much time to learn a new data about signed up member at the same time as the membership registration.
</details>

(The figure below is the create account page.)  
![화면 캡처 2022-05-29 162121](https://user-images.githubusercontent.com/96279383/170857028-9cdb1b92-5e6f-4b4b-8ad0-39a27921b3aa.png)

So, the book is recommended as shown in the picture below.
![화면 캡처 2022-05-29 162503](https://user-images.githubusercontent.com/96279383/177165927-1d29ca4e-ed31-4ff9-9771-ccece4ba05f3.png)

Finally, the user continues to log as they are active, and it is added back to <a href="ALS_model.py">the ALS recommendation model,</a> 
It will be retrained, and the trained <a href="ALS_model_serving.py">model is saved and served</a> again, and recommendations are made through aws lambda service. MLOps is implemented through repetition of this.

<details>
  <summary><b>Why use pyspark?</b></summary>
It was used to process large amounts of data. This article used only 100,000 data, but originally intended to use 50 million data. We succeeded up to 650,000 data
</details>


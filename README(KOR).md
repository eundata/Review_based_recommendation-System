# 리뷰기반 추천시스템 (엔코어 최종 프로젝트 22.04.06 ~ 22.04.29)

## 개요
우리는 아마존에서 판매된 책들의 리뷰데이터를 바탕으로 추천 시스템을 만들고자 했습니다. 추천 시스템에 리뷰 점수를 추가한 이유는 리뷰 점수의 융합이 정확도를 높인다는 여러 <a href="https://github.com/eundata/Recommendation-System/blob/main/papaers.md">논문</a>의 증거를 기반으로 했습니다. 
웹페이지에서 사용자 활동으로 인해 도서에 대한 평가 기록이나 리뷰 데이터가 누적되고나면, 매일 밤 모델을 재학습한 후 새로운 모델을 제공합니다. 이렇게 MLOPS를 구현해 보았습니다.
저는 이 프로젝트에서 추천 모델을 구축하는 일을 담당했습니다. (모든 과정은 aws에서 진행되었습니다)  
이 문서에서는 제가 수행한 부분만 설명이 있습니다.  

(아래그림은 데이터 흐름도와 데이터 구성요소입니다.)  
![화면 캡처 2022-05-25 140542](https://user-images.githubusercontent.com/96279383/170183938-9f9af045-8b36-4eec-9ce0-b9de168f2780.png)
![화면 캡처 2022-05-27 175313](https://user-images.githubusercontent.com/96279383/170814511-f4d17dda-1c29-4540-a666-293da8e98168.png)


## 방법
- 리뷰 데이터는 BERT 모델을 이용한 감성 분석 후 추천 시스템에 적용됩니다.  
- 기본적인 추천은 협업필터링에서 MF의 ALS(Alternating Least Square) 방식을 사용할 것입니다.

첫번째로, <a href = "https://github.com/eundata/Recommendation-System/blob/main/BERT.py">버트</a>를 전이학습하여 리뷰에 대한 점수를 부여합니다. 점수는 1점부터 5점까지입니다. 그 후, 리뷰점수와 감성점수의 평균을 내어 <a href = "https://colab.research.google.com/drive/1c61kuUElz8g9N0YpC10Zp0OjYuQJqzHy?usp=sharing">최종점수</a>를 만듭니다. 그 후 최종점수를 통해 ALS모델이 학습됩니다.
<details>
  <summary><b>최종 점수 파일에서 모델과 csv 파일을 가져와야 하는 이유는 무엇입니까?</b></summary>

모델과 그에 따른 데이터의 버전 관리를 위해 구축되었습니다.
</details>

둘째, 회원가입을 통해 사용자들이 어떤 책에 관심을 갖고 있는지 정보를 얻습니다. 회원가입할 때 리뷰가 많은 30권의 도서 목록이 주어지고 그 중 가장 좋아하는 도서 3권을 선택하게 합니다. 그 후, <a href='https://github.com/eundata/Recommendation-System/blob/main/Cosine_Similarity.py'>코사인 유사도</a>를 통하여 사용자가 선택한 책과 가장 유사한 책을 추천하게 됩니다.
<details>
  <summary><b>왜 코사인 유사도를 이용하였나요?</b></summary>

우리는 ALS모델에서 사용자기반협업필터링(UBCF)를 사용하려 했습니다. 그러나 새로 등록하는 사용자이기에 정보가 없어 UBCF방식으로 추천할 수 없고 회원등록과 동시에 ALS모델을 학습시켜 추천하는건 시간적으로 촉박하다 판단하여 일단 코사인 유사도를 이용해 추천을 하게끔 구축했습니다.
</details>

(아래 그림은 회원가입 페이지입니다.)  
![화면 캡처 2022-05-29 162121](https://user-images.githubusercontent.com/96279383/170857028-9cdb1b92-5e6f-4b4b-8ad0-39a27921b3aa.png)

그렇게 추천된 책은 아래 그림과 나타나게 됩니다.
![화면 캡처 2022-05-29 162503](https://user-images.githubusercontent.com/96279383/177165927-1d29ca4e-ed31-4ff9-9771-ccece4ba05f3.png)

마지막으로 사용자의 활동은 계속 기록되어지고, 이것은 <a href="ALS_model.py">ALS추천 모델</a>에 추가 됩니다.
다시 학습이되고, 그리고 재학습된 <a href="ALS_model_serving.py">모델은 저장, 서빙</a>되어, 람다서비스를 통해 추천이 완료됩니다. 이것의 반복으로 MLOps를 구현하고자 했습니다.

<details>
  <summary><b>왜 pyspark를 이용하였나요</b></summary>
스파크는 대용량 데이터를 처리하는 데에 사용됩니다. 이 글에서는 10만개의 데이터가 사용되었으나, 목표로 했던 데이터는 5000만개였습니다. 하지만 시간부족으로 실패했고 65만개 까지는 성공했습니다.
</details>


# Review based recommendation System</br>(encore final project 22.04.06 ~ 22.04.29)

## Abstract
We wanted to create a recommendation system based on review data of books sold on Amazon. The reason for adding review scores to the recommendation system was based on the evidence in <a href="https://github.com/eundata/Recommendation-System/blob/main/papaers.md">several papers</a> that cnvergence of review scores increase accuracy. 
When rating records or review data for books are accumulated due to user activity, the model is retrained every night and then a new model is served. In this way, we tried to embody the MLOPS.  
I was responsible for building a recommendation model in this project. (All the process was done on aws)  
This article describes only the part I performed  

(The figure below is the data flow and dataframe)
![화면 캡처 2022-05-25 140542](https://user-images.githubusercontent.com/96279383/170183938-9f9af045-8b36-4eec-9ce0-b9de168f2780.png)
![화면 캡처 2022-05-27 175313](https://user-images.githubusercontent.com/96279383/170814511-f4d17dda-1c29-4540-a666-293da8e98168.png)


## Method
- Review data is applied to the recommendation system after sentiment analysis using the BERT model and scoring.  
- The general recommendation is to use the Alternating Least Square (ALS) Matrix Factorization in collaborative filtering.  

First, we use the <a href = "https://colab.research.google.com/drive/1j4qpfaBtCnoRwC_VTW5u0XFrMRjHdsDL?usp=sharing#scrollTo=VUcT9DxGkdEd">BERT</a> to give a score for the review. Scores range from 1 to 5 points. Then, after averaging the user-given score and review score, the final score is made. After that, the model is trained with the final score to make the ALS model.

Second, through membership registration, we obtained information of which books that users are interested in. In the membership registration, users are given a list of 30 books with many reviews and are asked to choose about 3 of their favorite books. Then, using the <a href='https://github.com/eundata/Recommendation-System/blob/main/Cosine_Similarity.py'>cosine similarity</a>, a book with a high similarity to the book selected by the user is recommended.  
<details>
  <summary><b>Why use cosine similarity?</b></summary>

We plan to use the ALS recommendation model for UBCF. However, the cosine similarity method was first adopted because the information on the registered data is not included in the current model and it takes too much time to learn a new data about signed up member at the same time as the membership registration.
</details>

(The figure below is the create account page.)  
![화면 캡처 2022-05-29 162121](https://user-images.githubusercontent.com/96279383/170857028-9cdb1b92-5e6f-4b4b-8ad0-39a27921b3aa.png)

Third, 

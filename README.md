# Review based recommendation System</br>(encore final project 22.04.06 ~ 22.04.29)

## Abstract
We wanted to create a recommendation system based on review data of books sold on Amazon. The reason for adding review scores to the recommendation system was based on the evidence in <a href="https://github.com/eundata/Recommendation-System/blob/main/papaers.md">several papers</a> that review scores increase accuracy. I was responsible for building a recommendation model in this project. (All the process was done on aws)  
This article describes only the part I performed  

(The figure below is the data flow and dataframe)
![화면 캡처 2022-05-25 140542](https://user-images.githubusercontent.com/96279383/170183938-9f9af045-8b36-4eec-9ce0-b9de168f2780.png)
![화면 캡처 2022-05-27 175313](https://user-images.githubusercontent.com/96279383/170814511-f4d17dda-1c29-4540-a666-293da8e98168.png)


## Method
- Review data is applied to the recommendation system after sentiment analysis using the BERT model and scoring.  
- The general recommendation is to use the Alternating Least Square (ALS) Matrix Factorization in collaborative filtering  

First, we use the <a href = "https://github.com/eundata/Recommendation-System/blob/main/BERT_Model.py">BERT</a> to give a score for the review. Scores are given as negative and positive.
Then, after averaging the user-given score and review score, the final score is made.

First, we obtained information of which books that users are interested in through membership registration. In the membership registration, users are given a list of 30 books with many reviews and are asked to choose about 5 of their favorite books. Then, using the <a href='https://github.com/eundata/Recommendation-System/blob/main/Cosine_Similarity.py'>cosine similarity</a>, a book with a high similarity to the book selected by the user is recommended.  

<details>
  <summary><b>Why use cosine similarity?</b></summary>

We plan to use the ALS recommendation model for UBCF. However, the cosine similarity method was first adopted because the information on the registered data is not included in the current model and it takes too much time to learn a new data about signed up member at the same time as the membership registration.
</details>



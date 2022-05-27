# Review based recommendation System</br>(encore final project 22.04.06 ~ 22.04.29)

## Abstract
We wanted to create a recommendation system based on review data of books sold on Amazon. The reason for adding review scores to the recommendation system was based on the evidence in <a href="https://github.com/eundata/Recommendation-System/blob/main/papaers.md">several papers</a> that review scores increase accuracy. I was responsible for building a recommendation model in this project. (All the process was done on aws)  
This article describes only the part I performed  

(The figure below is the data flow and tools)
![화면 캡처 2022-05-25 140542](https://user-images.githubusercontent.com/96279383/170183938-9f9af045-8b36-4eec-9ce0-b9de168f2780.png)

## Method
- The general recommendation is to use the Alternating Least Square (ALS) Matrix Factorization in collaborative filtering  
- Review data is applied to the recommendation system after sentiment analysis using the BERT model and scoring.  

First, we obtained information of which books that users are interested in through membership registration  
In the membership registration, users are given a list of 30 books with many reviews and are asked to choose about 5 of their favorite books.  
Then, the first recommendation is fulfilled using the cosine similarity method.  

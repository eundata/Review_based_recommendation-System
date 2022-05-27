# Review based recommendation System</br>(encore final project 22.04.06 ~ 22.04.29)

## Abstract
We wanted to create a recommendation system based on review data of books sold on Amazon.  
The reason for adding review scores to the recommendation system was based on the evidence in several papers that review scores increase accuracy.
[papers][1]
    [1]: http://www.google.com

<Referenced papers>
Chen, X. Lv, J. Gou.(2020). Personalized Recommendation Model:  An Online CommentSentiment Based AnalysisS.  
Youdong Yun, Danial Hooshyar, Jaechoon Jo, Heuiseok Lim.(2017). Developing a hybrid collaborative filtering recommendation system with opinion mining on purchase review
Hongyan Li, Jun He, Tingting Wang, Wenting Song, Xiaoyang Du. (2013). Combining user preferences and user opinions for accurate recommendation  

I was responsible for building a recommendation model in this project. (All the process was done on aws)  
This article describes only the part I performed  

(The figure below is the data flow and tools)
![화면 캡처 2022-05-25 140542](https://user-images.githubusercontent.com/96279383/170183938-9f9af045-8b36-4eec-9ce0-b9de168f2780.png)

## Method
First, we obtained information of which books that users are interested in through membership registration  
After giving them a list of the 30 best-selling books, ask them to choose about 5 books they are interested in.  
Then, the first recommendation is fulfilled using the cosine similarity method.

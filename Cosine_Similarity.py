import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# The following is coded to load a file in s3 of aws.
df = pd.read_json('Books.json' ,lines=True)
df_Data = pd.DataFrame(df, columns=['asin', 'overall', 'reviewerID', 'reviewText'])
df_Data.index=df_Data.index+1
df_Data.head(30)

# Books with less than 4 reviews are deleted. Because I think a book with too few reviews is meaningless.
for x in df_Data.asin.unique():
    y=df_Data[df_Data.asin == x].shape[0]
    if y <= 3:
        df_Data = df_Data.drop(df_Data[df_Data.asin == x].index)

df_Data.index = df_Data.reset_index(drop=False, inplace=False).index+1

# Convert the type to float32 to save memory and find the cosine similarity between users
df_Data_cosine = df_Data.pivot_table('overall', index='asin', columns='reviewerID')
df_Data_cosine.fillna(0, inplace=True)
df_Data_cosine = df_Data_cosine.astype('float32')
cosine_out = cosine_similarity(df_Data_cosine)
cosine_out

# Top 10 Books with High Cosine Similarity
similarity_rate_df = pd.DataFrame(
    data = cosine_out,
    index = df_Data_cosine.index,
    columns = df_Data_cosine.index)
def recommand_system_10(book_id):
    print("neighbor 10")
    a_book_id = similarity_rate_df[book_id].sort_values(ascending=False)[1:11]
    print(a_book_id)
    
# Recommendation result value for book 'B00005N7P0'   
recommand_system_10('B00005N7P0')

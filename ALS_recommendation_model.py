import random
import os
import numpy as np
import json
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import mean, col, split, regexp_extract, when, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.evaluation import RegressionEvaluator


# The following is coded to load a file in s3 of aws.

bucket='dice-books'
data_key = 'Books_data_2022_05_20.csv'
data_location = 's3://dice-books/Books_data_2022_05_20.csv'.format(bucket, data_key)
pandasdf = pd.read_csv(data_location, lines=True)
pandasdf = pandasdf[['reviewerID', 'asin', 'composite_score']]


spark = SparkSession.builder\
        .appName('recommender_system')\
        .getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
df = spark.createDataFrame(pandasdf)


stringIndexer = StringIndexer(inputCol='asin',
                             outputCol='asin_new')
stringIndexer_ID = StringIndexer(inputCol='reviewerID',
                             outputCol='reviewerID_new')
model = stringIndexer.fit(df)
indexed = model.transform(df)
model_ID = stringIndexer_ID.fit(indexed)
indexed = model_ID.transform(indexed)


for x in range(200,1000,20):
    rec = ALS(rank=x,
        maxIter=10,
         regParam=0.01,
         userCol='reviewerID_new',
         itemCol='asin_new',
         ratingCol='composite_score',
         nonnegative=True,
         coldStartStrategy='drop')
    
    rec_model = rec.fit(train)
    pred_ratings = rec_model.transform(test)
    evaluator = RegressionEvaluator(labelCol='composite_score',
                                   predictionCol='prediction',
                                   metricName='rmse')
    rmse = evaluator.evaluate(pred_ratings)
    mae_eval = RegressionEvaluator(labelCol='composite_score',
                                  predictionCol='prediction',
                                  metricName='mae')
    mae = mae_eval.evaluate(pred_ratings)
    
    print(f"rank -{x}")
    print("RMSE:", rmse)
    print("MAE:", mae)
    


train, test = indexed.randomSplit([0.75, 0.25], 10)

rec = ALS(rank=200,
        maxIter=10,
         regParam=0.01,
         userCol='reviewerID_new',
         itemCol='asin_new',
         ratingCol='composite_score', # label -> predict할 때는 필요 없음!
         nonnegative=True,
         coldStartStrategy='drop')
"""
ALS(
    maxIter = doc='max number of iterations (>= 0).'
    regParam = doc='regularization parameter (>= 0).'
    userCol = doc='column name for user ids. Ids must be within the integer value range.'
    itemCol = doc='column name for item ids. Ids must be within the integer value range.'
    ratingCol = doc='column name for ratings'
    nonnegative = doc='whether to use nonnegative constraint for least squares'
    coldStartStrategy = doc="strategy for dealing with unknown or new users/items at prediction time. This may be useful in cross-validation or production scenarios, for handling user/item ids the model has not seen in the training data. Supported values: 'nan', 'drop'."
    )
"""
rec_model = rec.fit(train)
pred_ratings = rec_model.transform(test)
evaluator = RegressionEvaluator(labelCol='composite_score',
                               predictionCol='prediction',
                               metricName='rmse')
rmse = evaluator.evaluate(pred_ratings)
mae_eval = RegressionEvaluator(labelCol='composite_score',
                              predictionCol='prediction',
                              metricName='mae')
mae = mae_eval.evaluate(pred_ratings)

print("RMSE:", rmse)
print("MAE:", mae)


unique_asin = indexed.select("asin_new").distinct()

def top_movies(reviewerID_new, n):
    a = unique_asin.alias('a')
    user_buy = indexed.filter(indexed['reviewerID_new'] == reviewerID_new)\
                            .select('asin_new')
    b = user_buy.alias('b')
    total_movies = a.join(b, a['asin_new'] == b['asin_new'],
                         how='left')
    remaining_movies = total_movies\
                       .where(col('b.asin_new').isNull())\
                       .select('a.asin_new').distinct()
    remaining_movies = remaining_movies.withColumn('reviewerID_new',
                                                  lit(int(reviewerID_new)))
    recommender = rec_model.transform(remaining_movies)\
                           .orderBy('prediction', ascending=False)\
                           .limit(n)
    movie_title = IndexToString(inputCol='asin_new',
                               outputCol='title',
                               labels=model.labels)
    final_recommendations = movie_title.transform(recommender)
    
    return final_recommendations.show(n, truncate=False)

# sample user number is 7382.0
# top_movies(7382.0, 5)

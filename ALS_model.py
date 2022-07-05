import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import mean, col, split, regexp_extract, when, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

import os


# The code below is the process for setting up a spark session and connecting to aws.

os.environ['PYSPARK_SUBMIT_ARGS'] = '-- packages com.amazonaws:aws-java-sdk:1.7.4,org.apache.hadoop:hadoop-aws:2.7.3 pyspark-shell'

# spark configuration

conf = SparkConf().set('spark.executor.extraJavaOptions','-Dcom.amazonaws.services.s3.enableV4=true')\
 .set('spark.driver.extraJavaOptions','-Dcom.amazonaws.services.s3.enableV4=true')\
 .setAppName('pyspark_aws').setMaster('local[*]')
sc=SparkContext\
.getOrCreate(conf=conf)
sc.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')

accessKeyId='AKIA5WHQWSBCEPRHRW6P'
secretAccessKey='ENCOREDICE'

hadoopConf = sc._jsc.hadoopConfiguration()
hadoopConf.set('fs.s3a.access.key', accessKeyId)
hadoopConf.set('fs.s3a.secret.key', secretAccessKey)
hadoopConf.set('fs.s3a.endpoint', 's3-us-east-2.amazonaws.com')
hadoopConf.set('fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')
spark=SparkSession(sc)
SparkConf().get('spark.kryoserializer.buffer.max')
SparkConf().get('spark.driver.memory')


df=spark.read.json('s3://dice123/Magazine_Subscriptions.json')

# Index asin and reviewerID

stringIndexer = StringIndexer(inputCol='asin',
                             outputCol='asin_new')
stringIndexer_ID = StringIndexer(inputCol='reviewerID',
                             outputCol='reviewerID_new')

model_ID = stringIndexer_ID.fit(df)
indexed = model_ID.transform(df)
model = stringIndexer.fit(indexed)
indexed = model.transform(indexed)

# Indexing takes much time and it is required for the model serving part, so save the results.
model.save('s3://dice123/string_Indexer')


train, test = indexed.randomSplit([0.75, 0.25], 10)

rec = ALS(rank=200,
        maxIter=10,
         regParam=0.01,
         userCol='reviewerID_new',
         itemCol='asin_new',
         ratingCol='overall',
         nonnegative=True,
         coldStartStrategy='drop')

rec_model = rec.fit(train)
pred_ratings = rec_model.transform(test)


# rec_model is also required for the model serving part, so save it.
rec_model.save('s3://dice123/import_model')

# Save the dataframe in which the string is indexed into a csv file. it is also required for the model serving part.
indexed.write.option("header","false").csv('s3://dice123/indexed2')


for x in range(200,300,20):
    rec = ALS(rank=x,
        maxIter=10,
         regParam=0.01,
         userCol='reviewerID_new',
         itemCol='asin_new',
         ratingCol='overall',
         nonnegative=True,
         coldStartStrategy='drop')
    
    rec_model = rec.fit(train)
    pred_ratings = rec_model.transform(test)
    evaluator = RegressionEvaluator(labelCol='overall',
                                   predictionCol='prediction',
                                   metricName='rmse')
    rmse = evaluator.evaluate(pred_ratings)
    mae_eval = RegressionEvaluator(labelCol='overall',
                                  predictionCol='prediction',
                                  metricName='mae')
    mae = mae_eval.evaluate(pred_ratings)
    print(f"rank -{x}")
    print("RMSE:", rmse)
    print("MAE:", mae)


evaluator = RegressionEvaluator(labelCol='overall',
                               predictionCol='prediction',
                               metricName='rmse')
rmse = evaluator.evaluate(pred_ratings)
mae_eval = RegressionEvaluator(labelCol='overall',
                              predictionCol='prediction',
                              metricName='mae')
mae = mae_eval.evaluate(pred_ratings)
print("RMSE:", rmse)
print("MAE:", mae)


unique_asin = indexed.select("asin_new").distinct()


def top_books(reviewerID_new, n):

    a = unique_asin.alias('a')
    user_buy = indexed.filter(indexed['reviewerID_new'] == reviewerID_new)\
                            .select('asin_new')
    
    b = user_buy.alias('b')
    total_books = a.join(b, a['asin_new'] == b['asin_new'],
                         how='left')
    remaining_books = total_books\
                       .where(col('b.asin_new').isNull())\
                       .select('a.asin_new').distinct()
    remaining_books = remaining_books.withColumn('reviewerID_new',
                                                  lit(int(reviewerID_new)))
    recommender = rec_model.transform(remaining_books)\
                           .orderBy('prediction', ascending=False)\
                           .limit(n)
    books_title = IndexToString(inputCol='asin_new',
                               outputCol='title',
                               labels=model.labels)
    final_recommendations = books_title.transform(recommender)

    return final_recommendations.show(n, truncate=False)

# Recommend top 5 books for users with reviewerID_new of 139181.0
top_books(139181.0, 5)

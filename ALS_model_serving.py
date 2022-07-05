import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import mean, col, split, regexp_extract, when, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString, StringIndexerModel
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.types import DoubleType


rec_model = ALSModel.load('s3://dice123/import_model/')
model = StringIndexerModel.load('s3://dice123/string_Indexer')
indexed = spark.read.csv("s3://dice123/indexed2/")


indexed = indexed.selectExpr("_c0 as asin", "_c1 as overall", "_c2 as reviewerID", "_c3 as reviewerID_new", "_c4 as asin_new")
indexed = indexed.withColumn("reviewerID_new", indexed["reviewerID_new"].cast("double"))
indexed = indexed.withColumn("asin_new", indexed["asin_new"].cast("double"))
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

# top_books(139181.0, 5)

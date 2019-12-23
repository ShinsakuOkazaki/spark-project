from __future__ import print_function

import re
import sys
import numpy as np
from operator import add
import time
from pyspark import SparkContext
from operator import add

from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.mllib.classification import SVMWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
spark = SparkSession.builder\
    .master("local") \
    .appName("Word Count") \
    .getOrCreate()


def getFeature(x, category_map):
    ret = [0] * len(category_map)
    for e in x:
        ret[category_map.index(e)] = 1
    return ret

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)
    
    # Read file as spark dataframe.
    movies_meta = spark.read.csv(sys.argv[1], header=True)
    # Change datatype of column.
    movies_meta = movies_meta.withColumn('num_critic_for_reviews', movies_meta['num_critic_for_reviews'].cast("int"))\
                            .withColumn('duration', movies_meta['duration'].cast("float"))\
                            .withColumn('director_facebook_likes', movies_meta['director_facebook_likes'].cast('int'))\
                            .withColumn('actor_3_facebook_likes', movies_meta['actor_3_facebook_likes'].cast('int'))\
                            .withColumn('actor_1_facebook_likes', movies_meta['actor_1_facebook_likes'].cast('int'))\
                            .withColumn('gross', movies_meta['gross'].cast('float'))\
                            .withColumn('num_voted_users', movies_meta['num_voted_users'].cast('int'))\
                            .withColumn('cast_total_facebook_likes', movies_meta['cast_total_facebook_likes'].cast("int"))\
                            .withColumn('facenumber_in_poster', movies_meta['facenumber_in_poster'].cast('float'))\
                            .withColumn('num_user_for_reviews', movies_meta['num_user_for_reviews'].cast('int'))\
                            .withColumn('budget', movies_meta['budget'].cast('float'))\
                            .withColumn('title_year', movies_meta['title_year'].cast('int'))\
                            .withColumn('actor_2_facebook_likes', movies_meta['actor_2_facebook_likes'].cast('int'))\
                            .withColumn('imdb_score', movies_meta['imdb_score'].cast('float'))\
                            .withColumn('aspect_ratio', movies_meta['aspect_ratio'].cast('float'))\
                            .withColumn('movie_facebook_likes', movies_meta['movie_facebook_likes'].cast('int'))

    # Remove nan.
    without_na = movies_meta.na.drop()
    # Create target variable.
    without_na = without_na.withColumn('target', when(movies_meta['imdb_score'] >= 7.5, 1).otherwise(0))

    # Do label-encoding and one-hot-encoding.
    labels = ["color", "director_name", "actor_2_name", "actor_1_name", "actor_3_name",'language', "country" ,"content_rating"]
    for l in labels:
        indexer = StringIndexer(inputCol=l, outputCol=l + "_label")
        without_na= indexer.fit(without_na).transform(without_na)

    # Split genre elements.
    sep_str = (lambda x: x.split("|"))
    sep_str_udf = udf(sep_str, ArrayType(StringType()))
    genre_arr  = without_na.withColumn("genere_arr", sep_str_udf('genres'))
    

    data = genre_arr.select('target',
            'num_critic_for_reviews',
            'duration',
            'director_facebook_likes',
            'actor_3_facebook_likes',
            'actor_1_facebook_likes',
            'gross',
            'num_voted_users',
            'cast_total_facebook_likes',
            'facenumber_in_poster',
            'num_user_for_reviews',
            'budget',
            'title_year',
            'actor_2_facebook_likes',
            'aspect_ratio',
            'movie_facebook_likes',
            'color_label',
            'director_name_label',
            'actor_2_name_label',
            'actor_1_name_label',
            'actor_3_name_label',
            'language_label',
            'country_label',
            'content_rating_label',
            'genere_arr')

    # Convert dataframe to rdd.
    # This is inefficient, but for some operation rdd is preferable in this case.
    # This needs to be fixed later work.
    rdd = data.rdd.map(tuple)

    
    # Create features for each genre.
    seqOp = (lambda x1, x2: x1.union(x2))
    comOp = (lambda x1, x2: x1.union(x2))
    category_map = rdd.map(lambda x: (1, x[len(x)-1])).aggregateByKey(set(), seqOp, comOp).collect()[0][1]
    category_map = list(category_map)
    data_set = rdd.map(lambda x: (*x[:len(x)-1], *getFeature(x[len(x)-1], category_map)))

    # Split train and test data.
    train, test = data_set.randomSplit([0.9, 0.1], seed=12345)
    train_data = train.map(lambda x: LabeledPoint(x[0], np.array(x[1:])))

    # Train SVM.
    svm = SVMWithSGD.train(train_data, iterations=100, regParam=0.0)
    # Predict.
    predictionAndLabels_svm = test.map(lambda x: (float(svm.predict(x[1:])),float(x[0])))
    # Calculate metric.
    metrics_svm = BinaryClassificationMetrics(predictionAndLabels_svm)
    metrics2_svm = MulticlassMetrics(predictionAndLabels_svm)

    # Train LR.
    logistic = LogisticRegressionWithLBFGS.train(train_data, iterations=100)
    # Predict.
    predictionAndLabels_logistic = test.map(lambda x: (float(logistic.predict(x[1:])),float(x[0])))
    # Calculate metric.
    metrics_logistic = BinaryClassificationMetrics(predictionAndLabels_logistic)
    metrics2_logistic = MulticlassMetrics(predictionAndLabels_logistic)

    # Print results.
    print("Area under ROC for SVM = %s" % metrics_svm.areaUnderROC)
    print("Recall for SVM = %s" % metrics2_svm.recall())
    print("F1 Score for SVM = %s" % metrics2_svm.fMeasure())

    print("Area under ROC for LR = %s" % metrics_logistic.areaUnderROC)
    print("Precision for LR = %s" % metrics2_logistic.precision())
    print("Recall for LR = %s" % metrics2_logistic.recall())
    print("F1 Score for LR = %s" % metrics2_logistic.fMeasure())
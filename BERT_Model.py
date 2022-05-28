import os
import tensorflow as tf
from tensorflow import keras
import ktrain
from ktrain import text
import pandas as pd
import boto3
from tensorflow import keras
from keras.models import load_model
from keras_bert import get_custom_objects

# The following is coded to load a file in s3 of aws.
bucket='dice123'
data_key = 'Books.json'
data_location = 's3://dice123/Magazine_Subscriptions.json'.format(bucket, data_key)
df = pd.read_json(data_location, lines=True)
df1 = pd.DataFrame(df)
data_Tt=df1[['reviewText','overall']]

# 
Re = pd.DataFrame(data_Tt)
sentiment = {1: 'nagative',2: 'nagative',3: 'nagative',4: 'positive',5: 'positive'}
Re['sentiment']=Re['overall'].map(sentiment)
df2 = Re[['reviewText','sentiment']]

(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(train_df = d2, 
                                                                   text_column = 'reviewText',
                                                                   label_columns=['sentiment'],
                                                                   maxlen=100, 
                                                                   max_features=100000,
                                                                   preprocess_mode='bert',
                                                                   val_pct=0.1)

model = text.text_classifier(name='bert', train_data = (x_train, y_train) , preproc=preproc, metrics=['accuracy'])

learner = ktrain.get_learner(model = model, 
                             train_data=(x_train, y_train), 
                             val_data=(x_test, y_test), 
                             batch_size=32, 
                             use_multiprocessing = True)

# save model to s3 of aws
client_s3 = boto3.client("s3")
client_s3.download_file("dice123",'Ktrain_Bert_Model.h5', "./Ktrain_Bert_Model.h5")

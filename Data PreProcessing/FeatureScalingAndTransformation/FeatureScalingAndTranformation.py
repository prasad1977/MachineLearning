# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:38:51 2022

@author: Prasad
"""


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import math

#from utils import *
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

np.set_printoptions(precision=3)
diabetic_data= pd.read_csv('diabetes_processed.csv')
print(diabetic_data.head())

features_df = diabetic_data.drop('Outcome', axis=1)
target_df = diabetic_data['Outcome']
print(features_df.head())
print("#########Features Describe:#######\n",features_df.describe())


#Data to fit between 0 and 1, check the min and max after fit_transform

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_features = scaler.fit_transform(features_df)
print(rescaled_features[0:5])

rescaled_features_df = pd.DataFrame(rescaled_features, columns=features_df.columns)
print(rescaled_features_df.describe())


rescaled_features_df.boxplot(figsize=(10,7), rot=45)
print(plt.show())

####Data Standardize the data
###Calculate mean, then subscrtract from each feature value and divide by standard deviation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(features_df)
standardized_features = scaler.transform(features_df)
print(standardized_features[0:5])
standardized_features_df = pd.DataFrame(standardized_features, columns=features_df.columns)
print(standardized_features_df.describe())
standardized_features_df.boxplot(figsize=(10,7), rot=45)
print(plt.show())

###Data normalization
from sklearn.preprocessing import Normalizer
normlizer = Normalizer(norm='l1') # L2 or max
normlized_features = normlizer.fit_transform(features_df)
l1_normlized_features_df =pd.DataFrame(normlized_features, columns=features_df.columns)
print(l1_normlized_features_df.iloc[0])
print(l1_normlized_features_df.iloc[0].abs().sum())



####Binarizer
#### continuous value into categorical value
#### number of pregnancies above the mean is 1 and below the mean is 0

from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=float((features_df[['Pregnancies']]).mean()))
binarizer_features = binarizer.fit_transform(features_df[['Pregnancies']])
print(binarizer_features[0:10])

for i in range(1, features_df.shape[1]):
    scaler = Binarizer(threshold=float((features_df[[features_df.columns[i]]]).mean())).fit(features_df[[features_df.columns[i]]])

new_binarizer_feature = scaler.transform(features_df[[features_df.columns[i]]])

binarizer_features= np.concatenate((binarizer_features, new_binarizer_feature), axis=1)

print(binarizer_features[0:10])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def build_model (X , Y, test_frac):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)
    model = LogisticRegression(solver = 'liblinear').fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Test score: ", accuracy_score(y_test, y_pred))
    
print("####rescaled_features####")
build_model(rescaled_features, target_df, 0.2)
print("####standardized_features####")
build_model(standardized_features_df, target_df, 0.2)

print("####normalized_features####")
build_model(normlized_features, target_df, 0.2)

print("####binarizer_features####")
build_model(binarizer_features, target_df, 0.2)










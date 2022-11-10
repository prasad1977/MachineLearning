# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 18:07:39 2022

@author: Prasad
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

automobile = pd.read_csv('cars_processed.csv')
print(automobile.head())
print(automobile.describe())

automobile[['cylinders']] = preprocessing.scale(automobile[['cylinders']].astype('float64'))
automobile[['displacement']] = preprocessing.scale(automobile[['displacement']].astype('float64'))
automobile[['horsepower']] = preprocessing.scale(automobile[['horsepower']].astype('float64'))
automobile[['weight']] = preprocessing.scale(automobile[['weight']].astype('float64'))
automobile[['acceleration']] = preprocessing.scale(automobile[['acceleration']].astype('float64'))
automobile[['age']] = preprocessing.scale(automobile[['age']].astype('float64'))
print(automobile.describe())
print(automobile.shape)
from sklearn.model_selection import train_test_split
X= automobile.drop(['mpg', 'origin'], axis=1)
Y = automobile['mpg']

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.2)
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
lm=linear_model.fit(x_train, y_train)
print("Training Score: ", lm.score(x_train, y_train))

y_pred = lm.predict(x_test)
from sklearn.metrics import r2_score
print("Testing score: ", r2_score(y_test, y_pred))

#adjusted r2_score
def adjusted_r2(r_square, labels, features):
    adjusted_r_square = 1- ((1 - r_square) * (len(labels)-1 )) / (len(labels) - features.shape[1] -1)
    return adjusted_r_square

print("Adjusted_r2_score : ", adjusted_r2(r2_score(y_test, y_pred), y_test, x_test) )

features_corr = X.corr()
print(features_corr)
print(abs(features_corr)>0.8)


trimmed_features_df = X.drop(['cylinders','displacement', 'weight'], axis=1)
trimmed_features_corr = trimmed_features_df.corr()
print("trimmed_features_corr: \n", trimmed_features_corr)



















# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 09:31:31 2022

@author: Prasad
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

#from utils import *
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

auto_mobile_data= pd.read_csv('auto-mpg_processed.csv')
print(auto_mobile_data.head(10))
print(auto_mobile_data.shape)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#X = auto_mobile_data[['horsepower']]
#Y = auto_mobile_data[['mpg']]

X = auto_mobile_data.drop('mpg', axis=1)
Y = auto_mobile_data['mpg']
X_train, X_test, Y_train, Y_test = train_test_split (X,Y, test_size=0.2)

linear_model = LinearRegression (normalize=True).fit(X_train, Y_train)
print("#######Training Score:", linear_model.score(X_train, Y_train))

y_pred = linear_model.predict (X_test)
print("#######Testing Score:", r2_score(Y_test, y_pred))









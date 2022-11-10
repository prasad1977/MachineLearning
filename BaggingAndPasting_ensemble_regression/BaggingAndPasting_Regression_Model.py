# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 09:44:12 2022

@author: Prasad
"""
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
insur_data= pd.read_csv('insurance_processed.csv')
print(insur_data.head(10))
print(insur_data.shape)

X = insur_data.drop('charges', axis=1)
Y = insur_data['charges']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=0.2)
print(" Train Data:  ",X_train.shape,  Y_train.shape )
print(" Test Data  ",X_test.shape, "   ", Y_test.shape )

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score
"""
#with bootstraps=True means sampling with pasting
bag_reg = BaggingRegressor(DecisionTreeRegressor(), 
                           n_estimators=500,
                           bootstrap=True,
                           max_samples=0.8,
                           oob_score=True).fit(X_train,Y_train)
print("bag_reg.oob_score_:", bag_reg.oob_score_)
"""
#with bootstraps=False means sampling with pasting
bag_reg = BaggingRegressor(DecisionTreeRegressor(), 
                           n_estimators=500,
                           bootstrap=False,
                           max_samples=0.9
                           ).fit(X_train,Y_train)


y_pred = bag_reg.predict(X_test)
print("r2 score:", r2_score(Y_test, y_pred))













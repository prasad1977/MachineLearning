# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:06:45 2022

@author: Prasad
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

auto_data = pd.read_csv('cars_processed.csv')
print(auto_data.head())
print(auto_data.describe())

X = auto_data.drop(['mpg', 'origin'], axis=1)
Y = auto_data['mpg']

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.8)
lasso.fit(X,Y)

predictors = X.columns
coef = pd.Series(lasso.coef_, predictors).sort_values()
print(coef)

Lasso_features = ['age','weight']
print(X[Lasso_features].head())

from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor(max_depth=4)
decision_tree.fit(X,Y)
predictors = X.columns
coef = pd.Series(decision_tree.feature_importances_, predictors).sort_values()
print(coef)
decision_tree_features = ['displacement','horsepower' ]
print(X[decision_tree_features].head())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def build_model(X,Y, test_frac):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)
    
    model = LogisticRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print("Test Score:", r2_score(y_test, y_pred))


build_model(X[Lasso_features], Y, 0.2)
build_model(X[decision_tree_features], Y, 0.2)





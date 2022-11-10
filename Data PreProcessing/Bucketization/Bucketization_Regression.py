# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:36:42 2022

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

X = auto_mobile_data[['horsepower']]
Y = auto_mobile_data[['mpg']]

plt.figure(figsize=(10,8))
plt.plot(X, Y, 'o', c='y')
print(plt.show())
print("#####Graph completed#####")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
lm = LinearRegression()
model = lm.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print("r2_score: ", r2_score(Y_test, y_pred))
plt.figure(figsize=(10,8))
plt.plot(X_train, Y_train, 'o', c='y')
plt.plot(X_test, y_pred, linewidth=2, color='green', linestyle='-',
         label='linear regression')
print(plt.show())
print("#####Graph completed#####")

from sklearn.preprocessing import KBinsDiscretizer

encoder = KBinsDiscretizer(n_bins=20, encode='ordinal')
x_binned = encoder.fit_transform(X_train)
print(x_binned[:10])
x_test_binned=encoder.transform(X_test)

reg = LinearRegression().fit(x_binned, Y_train)
z_pred = reg.predict(x_test_binned)
print(z_pred)
print("r2_score: ", r2_score(Y_test,z_pred))



# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:45:07 2022

@author: Prasad
"""

import numpy
import pandas
import warnings
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
import pickle


from matplotlib import pyplot as plt

pandas.set_option('display.max_rows',550)
pandas.set_option('display.max_columns',500)
pandas.set_option('display.width',500)

filename='forestfires.csv'

names=['X','Y','month',	'day',	'FFMC',	'DMC',	'DC',	'ISI',	'temp',	'RH',	'wind',	'rain',	'area']
df=pandas.read_csv(filename,names=names)
df.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)
df.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7),inplace=True)

"""
max_error_scoring='max_error'
neg_mean_absolute_error_scoring='neg_mean_absolute_error'
r2_scoring='r2'
neg_mean_squared_error_scoring='neg_mean_squared_error'

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO'), Lasso())
models.append(('EN'), ElasticNet())
models.append(('Ridge'), Ridge())
models.append(('KNN'), KNeighborsRegressor())
models.append(('CART'), DecisionTreeRegressor())
models.append(('SVR'), SVR())
"""

df['X'] = pandas.to_numeric(df['X'], errors='coerce').fillna(0).astype(int)
df['Y'] = pandas.to_numeric(df['Y'], errors='coerce').fillna(0).astype(int)
df['month'] = pandas.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
df['day'] = pandas.to_numeric(df['day'], errors='coerce').fillna(0).astype(int)
df['FFMC'] = pandas.to_numeric(df['FFMC'], errors='coerce').fillna(0).astype(int)
df['DMC'] = pandas.to_numeric(df['DMC'], errors='coerce').fillna(0).astype(int)
df['DC'] = pandas.to_numeric(df['DC'], errors='coerce').fillna(0).astype(int)
df['ISI'] = pandas.to_numeric(df['ISI'], errors='coerce').fillna(0).astype(int)
df['temp'] = pandas.to_numeric(df['temp'], errors='coerce').fillna(0).astype(int)
df['RH'] = pandas.to_numeric(df['RH'], errors='coerce').fillna(0).astype(int)
df['wind'] = pandas.to_numeric(df['wind'], errors='coerce').fillna(0).astype(int)
df['rain'] = pandas.to_numeric(df['rain'], errors='coerce').fillna(0).astype(int)
df['area'] = pandas.to_numeric(df['area'], errors='coerce').fillna(0).astype(int)

array=df.values
X=array[:,0:12]
Y=array[:,12]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=1,shuffle=True)



lasso_model=Lasso()
lasso_model.fit(X_train,Y_train)

predictions= lasso_model.predict(X_test)
print(predictions)

print('Predictions complete')
pickle.dump(lasso_model,open('model.pkl','wb'))
print('model dump complete')




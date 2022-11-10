# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:39:33 2022

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

auto_mobile_data= pd.read_csv('auto-mpg.csv')
print(auto_mobile_data['year'].str.isnumeric().value_counts())
print(auto_mobile_data['year'].loc[auto_mobile_data['year'].str.isnumeric() == False])
extr = auto_mobile_data['year'].str.extract(r'^(\d{4})', expand=False)

print(extr.head())
auto_mobile_data['year'] = pd.to_numeric(extr[0][-2:])
auto_mobile_data['year'] = pd.to_numeric(extr[1][-2:])

print(extr.head())

auto_mobile_data['year'].dtype
print("##########END oF YEAR######")

print(auto_mobile_data['cylinders'].str.isnumeric().value_counts())
print(auto_mobile_data['cylinders'].loc[auto_mobile_data['cylinders'].str.isnumeric()== False])
auto_mobile_data['cylinders']=auto_mobile_data['cylinders'].replace('_', np.mean(pd.to_numeric(auto_mobile_data['cylinders'], errors='coerce'))).astype(int)

print("After replace _ with mean",auto_mobile_data['cylinders'].dtype)

print("########END oF cylinders########")
auto_mobile_data['displacement'] = pd.to_numeric(auto_mobile_data['displacement'], errors='coerce')
auto_mobile_data['weight'] = pd.to_numeric(auto_mobile_data['weight'], errors='coerce')

print(auto_mobile_data['displacement'].dtype)
print("########END oF displacement########")
auto_mobile_data['mpg'] = pd.to_numeric(auto_mobile_data['mpg'], errors='coerce').fillna(0).astype(int)
#auto_mobile_data['year'] = pd.to_numeric(auto_mobile_data['year'], errors='coerce').fillna(0).astype(str)


print(auto_mobile_data.head())

auto_mobile_data = auto_mobile_data.replace('?',np.nan)
print("####NAN values:\n", auto_mobile_data.isna().sum())
auto_mobile_data['mpg'] = auto_mobile_data['mpg'].fillna(auto_mobile_data['mpg'].mean())
print("####NAN values:\n", auto_mobile_data.isna().sum())
auto_mobile_data = auto_mobile_data.dropna()
print(auto_mobile_data.shape)
print("####NAN values:\n", auto_mobile_data.isnull().sum())

auto_mobile_data = auto_mobile_data.drop('Model', axis=1)

print(auto_mobile_data.head(10))

auto_mobile_data.to_csv('auto-mpg_processed.csv', index=False)


# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:17:01 2022

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

plt.figure(figsize=(12, 8))
plt.scatter(auto_mobile_data['acceleration'], auto_mobile_data['mpg'], color='g')
plt.xlabel('acceleration')
plt.ylabel('mpg')
print(plt.show())

plt.figure(figsize=(12, 8))
plt.scatter(auto_mobile_data['weight'], auto_mobile_data['mpg'], color='r')
plt.xlabel('weight')
plt.ylabel('mpg')
print(plt.show())


auto_mobile_data.plot.scatter(x='weight',
                              y='acceleration',
                              c='horsepower',
                              colormap='viridis',
                              figsize=(12,8))
print(plt.show())

plt.bar(auto_mobile_data['cylinders'], auto_mobile_data['mpg'])
plt.xlabel('cylinders')
plt.ylabel('mpg')
print(plt.show())


cars_corr= auto_mobile_data.corr()
print("########Correlation#########\n",cars_corr)

fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(cars_corr, annot=True)

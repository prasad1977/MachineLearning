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
insur_data= pd.read_csv('insurance.csv')

print("Sample insure Data\n", insur_data.head(10))
print("Shape of the insur data\n", insur_data.shape)
print("Stats of insur data\n", insur_data.describe())

insur_corr = insur_data.corr()
print("inusr data correlation...\n", insur_corr)

fig, ax = plt.subplots(figsize=(9,9))
sns.heatmap(insur_corr, annot=True)

label_encoding = preprocessing.LabelEncoder()
insur_data['region'] = label_encoding.fit_transform(insur_data['region'].astype(str))
print(insur_data.head())
print(label_encoding.classes_)
insur_data = pd.get_dummies(insur_data, columns=['sex','smoker'])

print(insur_data.head())

insur_data.to_csv('insurance_processed.csv', index=False)











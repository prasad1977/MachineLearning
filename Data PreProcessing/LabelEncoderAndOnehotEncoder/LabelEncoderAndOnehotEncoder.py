# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:57:31 2022

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
gosales= pd.read_csv('GoSales_Tx_LogisticRegression.csv')
print(gosales.head())
print(gosales.shape)
print(gosales.AGE.describe())

plt.figure(figsize=(10,7))
pd.value_counts(gosales['IS_TENT']).plot.bar()
print(plt.show())

plt.figure(figsize=(10,7))
pd.value_counts(gosales['MARITAL_STATUS']).plot.bar()
print(plt.show())


plt.figure(figsize=(10,7))
pd.value_counts(gosales['GENDER']).plot.bar()
print(plt.show())

plt.figure(figsize=(10,7))
pd.value_counts(gosales['PROFESSION']).plot.bar()
print(plt.show())


gender = ['M', 'F']
from sklearn import preprocessing
label_encoding = preprocessing.LabelEncoder()
label_encoding = label_encoding.fit(gender)
gosales['GENDER'] = label_encoding.transform(gosales['GENDER'].astype(str))
print(label_encoding.classes_)
print(gosales.sample(10))



print(gosales[['MARITAL_STATUS']].sample(5))

one_hot_encoder = preprocessing.OneHotEncoder()
one_hot_encoder = one_hot_encoder.fit(gosales['MARITAL_STATUS'].values.reshape(-1,1))
print(one_hot_encoder.categories_)

one_hot_labels = one_hot_encoder.transform(gosales['MARITAL_STATUS'].values.reshape(-1,1)).toarray()

print(one_hot_labels)
labels_df = pd.DataFrame()
labels_df['MARITAL_STATUS_Married'] = one_hot_labels[:,0]
labels_df['MARITAL_STATUS_single'] = one_hot_labels[:,1]
labels_df['MARITAL_STATUS_Unspecified'] = one_hot_labels[:,2]
print(labels_df.head(10))

encoded_df = pd.concat([gosales, labels_df], axis=1)
encoded_df.drop('MARITAL_STATUS', axis=1, inplace=True)
print(encoded_df.head(10))

gosales=pd.get_dummies(encoded_df, columns=['PROFESSION'])
print(gosales.sample(10))



###instead of all the above, we can call get_dummies for complete dataset
gosales= pd.read_csv('GoSales_Tx_LogisticRegression.csv')
gosales=pd.get_dummies(gosales)
print(gosales.sample(10))


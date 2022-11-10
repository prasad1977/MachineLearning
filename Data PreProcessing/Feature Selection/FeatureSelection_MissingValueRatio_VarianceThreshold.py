# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:16:08 2022

@author: Prasad
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

diabetic_data = pd.read_csv('diabetes.csv')
print(diabetic_data.head())
print(diabetic_data.describe())

print(diabetic_data.isnull().sum())
diabetic_data['Glucose'].replace(0, np.nan, inplace=True)
diabetic_data['Pregnancies'].replace(0, np.nan, inplace=True)
diabetic_data['BloodPressure'].replace(0, np.nan, inplace=True)
diabetic_data['BMI'].replace(0, np.nan, inplace=True)
diabetic_data['Insulin'].replace(0, np.nan, inplace=True)
diabetic_data['SkinThickness'].replace(0, np.nan, inplace=True)

print(diabetic_data.isnull().sum())
print("Diabetic missing zero %:", diabetic_data['Glucose'].isnull().sum()/ len(diabetic_data) * 100)
print("Diabetic missing zero %:", diabetic_data['BloodPressure'].isnull().sum()/ len(diabetic_data) * 100)
print("Diabetic missing zero %:", diabetic_data['SkinThickness'].isnull().sum()/ len(diabetic_data) * 100)
print("Diabetic missing zero %:", diabetic_data['Insulin'].isnull().sum()/ len(diabetic_data) * 100)



diabetic_data_processed = pd.read_csv('diabetes_processed.csv')
print(diabetic_data_processed.head())
X = diabetic_data_processed.drop('Outcome', axis=1)
Y = diabetic_data_processed['Outcome']
print("diabetic_data_processed variance...\n", X.var(axis=0))

from sklearn.preprocessing import minmax_scale
X_scaled = pd.DataFrame(minmax_scale(X, feature_range = (0,10)), 
                        columns=X.columns)

print("diabetic_data_processed variance...\n", X_scaled.var(axis=0))

from sklearn.feature_selection import VarianceThreshold

select_features = VarianceThreshold(threshold=1.0)

X_new = select_features.fit_transform(X_scaled)

print(X_new.shape)

X_new = pd.DataFrame(X_new)

print(X_new.head())
























# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 10:18:14 2022

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

diabetic_data= pd.read_csv('diabetes.csv')
print(diabetic_data.head())
print("Shape of the data:", diabetic_data.shape)
print("Info of the data:\n", diabetic_data.info())
print("Data Describe:\n")
print( diabetic_data.describe().transpose())
diabetic_data['Glucose'].replace(0,np.nan, inplace=True)
diabetic_data['BloodPressure'].replace(0,np.nan, inplace=True)
diabetic_data['SkinThickness'].replace(0,np.nan, inplace=True)
diabetic_data['Insulin'].replace(0,np.nan, inplace=True)
diabetic_data['BMI'].replace(0,np.nan, inplace=True)

print(diabetic_data.isnull().sum())

arr = diabetic_data['SkinThickness'].values.reshape(-1,1)
print("arr shape:", arr.shape)


from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
imp.fit(diabetic_data['SkinThickness'].values.reshape(-1,1))
diabetic_data['SkinThickness'] = imp.transform(diabetic_data['SkinThickness'].values.reshape(-1,1))
print("diabetic_data['SkinThickness']:", diabetic_data['SkinThickness'].describe())


print(diabetic_data.isnull().sum())

imp = SimpleImputer(missing_values=np.nan, strategy = 'median')
imp.fit(diabetic_data['Glucose'].values.reshape(-1,1))
diabetic_data['Glucose'] = imp.transform(diabetic_data['Glucose'].values.reshape(-1,1))
print("diabetic_data['Glucose']:", diabetic_data['Glucose'].describe())


print(diabetic_data.isnull().sum())

imp = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imp.fit(diabetic_data['BloodPressure'].values.reshape(-1,1))
diabetic_data['BloodPressure'] = imp.transform(diabetic_data['BloodPressure'].values.reshape(-1,1))
print("diabetic_data['Glucose']:", diabetic_data['BloodPressure'].describe())


imp = SimpleImputer(missing_values=np.nan, strategy = 'constant' , fill_value=32)
imp.fit(diabetic_data['BMI'].values.reshape(-1,1))
diabetic_data['BMI'] = imp.transform(diabetic_data['BMI'].values.reshape(-1,1))
print("diabetic_data['Glucose']:", diabetic_data['BMI'].describe())

print(diabetic_data.isnull().sum())

diabetic_data.to_csv('diabetes_processed_incomplete.csv', index=False)

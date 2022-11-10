# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 18:23:23 2022

@author: Prasad
"""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 


#from utils import *
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

imp= IterativeImputer(max_iter=100, random_state=0)
diabetic_data= pd.read_csv('diabetes_processed_incomplete.csv')
print(diabetic_data.head())
print(diabetic_data.isnull().sum())

diabetic_features = diabetic_data.drop('Outcome', axis = 1)
diabetis_label = diabetic_data[['Outcome']]
print(diabetic_features.head())


imp= IterativeImputer(max_iter=1000, random_state=0)
imp.fit(diabetic_features)
diabetic_features_arr = imp.transform(diabetic_features)
print(diabetic_features_arr.shape)
diabetic_features = pd.DataFrame(diabetic_features_arr, columns=diabetic_features.columns)
print(diabetic_features)
diabetic_data=pd.concat([diabetic_features,diabetis_label], axis=1)
print(diabetic_data.isnull().sum())
diabetic_data.to_csv('diabetes_processed.csv', index=False)











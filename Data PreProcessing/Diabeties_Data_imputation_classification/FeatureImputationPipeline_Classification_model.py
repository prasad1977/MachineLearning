# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 19:19:57 2022

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

diabetic_data= pd.read_csv('diabetes_processed.csv')
print(diabetic_data.head(10))

diabetic_features = diabetic_data.drop('Outcome', axis = 1)
diabetis_label = diabetic_data[['Outcome']]
#performing feature imputation and classification using a pipeline
#random imputation
mask = np.random.randint(0,100,size=diabetic_features.shape).astype(np.bool)
mask = np.logical_not(mask)
diabetic_features[mask] = np.nan
print("After Masking the data:\n", diabetic_features.sample(15))
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#x=diabetic_data.drop('Outcome', axis=1)
#y=diabetic_data['Outcome']
x_train,x_test,y_train, y_test = train_test_split(diabetic_features, diabetis_label, test_size=0.20)

transformer=ColumnTransformer(
    transformers=[('features', SimpleImputer(strategy='mean'), [0,1,2,3,4,5,6,7])]
    )

#clf = DecisionTreeClassifier(max_depth=4).fit(x_train,y_train)
clf = make_pipeline(transformer, DecisionTreeClassifier(max_depth=4))
clf = clf.fit(x_train,y_train)
print("clf.score()\n",clf.score(x_train, y_train))

y_pred=clf.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:01:59 2022

@author: Prasad
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

diabetes = pd.read_csv('diabetes.csv')
print(diabetes.head())

diabetes_corr = diabetes.corr()
print("#######Correlations#####\n",diabetes_corr)

plt.figure(figsize=(8,8))
sns.heatmap(diabetes_corr, annot=True)
print(plt.show())

#yellow brick library

X = diabetes[['Insulin', 'BMI', 'BloodPressure', 'DiabetesPedigreeFunction']]
Y = diabetes['Age']

feature_names = X.columns
print(feature_names)


from yellowbrick.target import FeatureCorrelation

visualizer = FeatureCorrelation( labels=feature_names, method='pearson')

visualizer.fit(X,Y)
visualizer.poof()
print(visualizer.scores_)
print(visualizer.features_)

score_df = pd.DataFrame({'Feature Names' : visualizer.features_,
                         'Scores' : visualizer.scores_})
## Pearson correlation is good only for continuous numerical varible
print(score_df)
X = diabetes.drop('Outcome', axis=1)
Y = diabetes['Outcome']
feature_names = X.columns
visualizer = FeatureCorrelation(lables=feature_names,method='pearson')
visualizer.fit(X,Y)
visualizer.poof()


### To calculate the correlation for discreate variables
###think Pregnancies is a discreate variable


discreate_features = [False for _ in range(len(feature_names))]
discreate_features[0] = True

visualizer = FeatureCorrelation(method='mutual_info-classification', 
                                labels = feature_names)
visualizer.fit(X,Y, discreate_features=discreate_features, random_state=0)
visualizer.poof()








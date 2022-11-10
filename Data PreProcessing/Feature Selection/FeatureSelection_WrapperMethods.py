# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:51:24 2022

@author: Prasad
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

diabetic_data = pd.read_csv('diabetes.csv')
print(diabetic_data.head())
print(diabetic_data.describe())
X = diabetic_data.drop('Outcome', axis=1)
Y = diabetic_data['Outcome']


# Recursive feature selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver = 'liblinear')
rfe = RFE(model, n_features_to_select=4)
fit = rfe.fit(X,Y)

print("Num Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_)

feature_rank = pd.DataFrame( {'columns': X.columns,
                            'ranking': fit.ranking_,
                            'selected': fit.support_})

print("feature_rank:\n", feature_rank)

recursive_feature_names= feature_rank.loc[feature_rank['selected'] == True ]
print("recursive_feature_ranks:\n", recursive_feature_names)    

X[recursive_feature_names['columns'].values].head()
print(X[recursive_feature_names['columns'].values].head())
                                      
recursive_features= X[recursive_feature_names['columns'].values]
                                  
# Sequential feature selection - Forward Direction

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
feature_selector = SequentialFeatureSelector(clf,
                                             n_features_to_select=4, direction='forward')
                                             
features = feature_selector.fit(np.array(X),Y)

forward_elimination_feature_names = list(X.columns[list(features.get_support())])
print("forward_elimination_feature_names: \n", forward_elimination_feature_names)


forward_elimination_features = X[forward_elimination_feature_names]


# Sequential feature selection - Backward Direction
backward_feature_selector = SequentialFeatureSelector(clf,
                                             n_features_to_select=4, direction='forward')
                                             
Backward_features = backward_feature_selector.fit(np.array(X),Y)

bakward_elimination_feature_names = list(X.columns[list(Backward_features.get_support())])

print("bakward_elimination_feature_names: \n", bakward_elimination_feature_names)


bakward_elimination_features=X[bakward_elimination_feature_names]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def build_model(X,Y, test_frac):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)
    
    model = LogisticRegression(solver='liblinear').fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print("Test Score:", accuracy_score(y_test, y_pred))


build_model(X, Y, 0.2)
build_model(forward_elimination_features, Y, 0.2)
build_model(bakward_elimination_features, Y, 0.2)



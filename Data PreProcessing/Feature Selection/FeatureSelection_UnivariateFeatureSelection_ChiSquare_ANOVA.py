# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:37:54 2022

@author: Prasad
"""
# building chi-square and ANOVA

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

def get_selected_features(X, X_new):
    selected_features = []
    
    for i in range(len(X_new.columns)):
        for j in range(len(X.columns)):
            if(X_new.iloc[:,i].equals(X.iloc[:,j])):
                print(X.columns[j])
                selected_features.append(X.columns[j])
            
    return selected_features

# implementing Chisquare
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

X = diabetic_data.drop('Outcome', axis=1)
Y = diabetic_data['Outcome']

print(X.shape)

X = X.astype(np.float64)
test = SelectKBest(score_func=chi2,k=4)
fit = test.fit(X, Y)

print("chi2 score:" , fit.scores_)

feature_score = pd.DataFrame()
for i in range(X.shape[1]):
    new = pd.DataFrame({'Features' : X.columns[i],
                        'Score' : fit.scores_[i]}, index =[i])
    
    feature_score = pd.concat([feature_score,new])

print("feature_score\n", feature_score)

X_new = fit.transform(X)
X_new = pd.DataFrame(X_new)
print(X_new.head())
selected_features = get_selected_features(X, X_new)
print(selected_features)
chi2_best_features = X[selected_features]


# implementing ANOVA
from sklearn.feature_selection import f_classif, SelectPercentile
test = SelectPercentile(f_classif, percentile=80)
test.fit(X,Y)

print("ANOVA scores:\n" , fit.scores_)
X_new = fit.transform(X)
X_new = pd.DataFrame(X_new)
print(X_new.head(10))

selected_features = get_selected_features(X, X_new)
print(selected_features)
anova_best_features = X[selected_features]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def build_model(X,Y, test_frac):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)
    
    model = LogisticRegression(solver='liblinear').fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print("Test Score:", accuracy_score(y_test, y_pred))


build_model(X, Y, 0.2)
build_model(chi2_best_features, Y, 0.2)
build_model(anova_best_features, Y, 0.2)






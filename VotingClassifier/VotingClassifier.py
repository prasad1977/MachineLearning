# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 20:17:31 2022

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

diabetis_data= pd.read_csv('diabetes_processed.csv')
print(diabetis_data.head(10))


X = diabetis_data.drop('Outcome', axis=1)
Y = diabetis_data['Outcome']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.20 )

print(X_train.shape, "  ", Y_test.shape )

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

log_clf = LogisticRegression(C=1, solver='liblinear')
svc_clf = SVC(C=1, kernel='linear',gamma = 'auto')
naive_clf = GaussianNB()

voting_clf_hard = VotingClassifier(estimators=[('lr' , log_clf),
                                               ('svc' , svc_clf),
                                               ('naive',naive_clf)],
                                    voting='hard')

voting_clf_hard.fit(X_train, Y_train)

VotingClassifier(estimators=[('lr' , LogisticRegression(C=1, class_weight=None,
                                                    dual=False,fit_intercept=True,
                                                    intercept_scaling = 1, l1_ratio= None,
                                                    max_iter=100, multi_class='warn',n_jobs=None,
                                                    penalty='12', random_state=None,
                                                    solver='liblinear', tol=0.0001, verbose=0,
                                                    warm_start=False)),
                                    ('svc', SVC(C=1,cache_size=200,class_weight=None, 
                                                coef0=0.0, decision_function_shape='ovr',
                                                degree=3, gamma='auto',kernel='linear',
                                                max_iter=-1, probability=False,random_state=None,
                                                shrinking=True,tol=0.001,verbose=False)),
                                    ('naive', GaussianNB(priors=None,var_smoothing=1e-09))],
                    flatten_transform=True, n_jobs=None,voting= 'hard',
                    weights=None)
                                               
                                    
y_pred = voting_clf_hard.predict(X_test)                    
                                    
from sklearn.metrics import accuracy_score, precision_score, recall_score
print(accuracy_score(Y_test, y_pred))
                                    
for clf_hard in (log_clf, svc_clf, naive_clf, voting_clf_hard):
    clf_hard.fit(X_train, Y_train)
    y_pred = clf_hard.predict(X_test)
    
    print(clf_hard.__class__.__name__, accuracy_score(Y_test, y_pred))
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    

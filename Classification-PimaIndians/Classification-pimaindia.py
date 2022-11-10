# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:09:27 2022

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

diabetis_data= pd.read_csv('diabetes.csv')
print(diabetis_data['Glucose'])
print(diabetis_data.head(10))
print(diabetis_data.shape)
print(diabetis_data.describe())

plt.Figure(figsize=(9,9))
plt.scatter(diabetis_data['Glucose'],diabetis_data['Outcome'],c='y')
plt.xlabel('Glucose')
plt.ylabel('Test')
print(plt.show())
plt.scatter(diabetis_data['Age'],diabetis_data['Insulin'],c='y')
print(plt.show())

plt.scatter(diabetis_data['Pregnancies'],diabetis_data['Insulin'],c='y')
print(plt.show())

#print(plt.show())
#plt.scatter(diabetis_data['TV'],diabetis_data['Sales'],c='y')
#print(plt.show())

diabetic_corr = diabetis_data.corr()
print( diabetic_corr)

fig, ax = plt.subplots(figsize=(9,9))
sns.heatmap(diabetic_corr, annot=True)

# Model starts

features=diabetis_data.drop('Outcome', axis=1)
from sklearn import preprocessing
standard_scaler = preprocessing.StandardScaler()

featured_scaled = standard_scaler.fit_transform(features)

print(featured_scaled.shape)
featured_scaled_df=pd.DataFrame(featured_scaled, columns=features.columns)
print(featured_scaled_df.head())
print(featured_scaled_df.describe())

diabetis_data= pd.concat([featured_scaled_df,diabetis_data['Outcome']], axis=1).reset_index(drop=True)

print(diabetis_data.head())
diabetis_data.to_csv('diabetes_processed.csv', index=False)

from sklearn.model_selection import train_test_split
X=diabetis_data.drop('Outcome', axis=1)
Y=diabetis_data['Outcome']
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.30 )

print(" Train Data:  ",X_train.shape,  Y_train.shape )
print(" Test Data  ",X_test.shape, "   ", X_test.shape )
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

print("Predictions", Y_pred)

pred_results = pd.DataFrame ( {"Y_test": Y_test, 'Y_pred': Y_pred})
print("Prediction results: ")
print(pred_results.head(10))


# Model accuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score
model_accuracy = accuracy_score(Y_test, Y_pred)
model_precision = precision_score(Y_test, Y_pred)
model_recall = recall_score(Y_test, Y_pred)
print("Accuracy of the model is {}% " .format(model_accuracy * 100))
print("Precision of the model is {}% " .format(model_precision * 100))
print("recall of the model is {}% " .format(model_recall * 100))



# build model using Decision Tree Classifier 
from sklearn.tree import DecisionTreeClassifier
classifier1 = DecisionTreeClassifier(max_depth=4)
classifier1.fit(X_train, Y_train)
Y_pred = classifier1.predict(X_test)
print("Y_pred using DecisionTreeClassifier: \n", Y_pred)
pred_results = pd.DataFrame ( {"Y_test": Y_test, 'Y_pred': Y_pred})
print("Prediction results DecisionTreeClassifier: \n" ,pred_results.head(10))

model_accuracy = accuracy_score(Y_test, Y_pred)
model_precision = precision_score(Y_test, Y_pred)
model_recall = recall_score(Y_test, Y_pred)
print("Accuracy of the model is {}% " .format(model_accuracy * 100))
print("Precision of the model is {}% " .format(model_precision * 100))
print("recall of the model is {}% " .format(model_recall * 100))

diabetes_crosstab = pd.crosstab(pred_results.Y_pred, pred_results.Y_test)
print("Confusion Matrix: \n " , diabetes_crosstab)

TP = diabetes_crosstab[1][1]
TN = diabetes_crosstab[0][0]
FP = diabetes_crosstab[0][1]
FN = diabetes_crosstab[1][0]

accuracy_score_verified = (TP + TN) / (TP + TN + FP + FN)
print("accuracy_score_verified:", accuracy_score_verified)
precision_score_verified  = TP / ( TP+ FP)
print("precision_score_verified:", precision_score_verified)
recall_score_verified = TP / ( TP+FN)
print("recall_score_verified:", recall_score_verified)


# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:09:41 2022

@author: Prasad
"""

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

diabetis_data= pd.read_csv('diabetes_processed.csv')
print(diabetis_data.head(10))
print(diabetis_data.shape)
print(diabetis_data.describe())

FEATURES=list(diabetis_data.columns[:-1])
print(FEATURES)


from sklearn.decomposition import PCA
def apply_pca(n):
    pca= PCA(n_components=(n))
    x_new=pca.fit_transform(diabetis_data[FEATURES])
    return pca, pd.DataFrame(x_new)

pca_obj, _ = apply_pca(8)

print("Explained Variance:" , pca_obj.explained_variance_ratio_)

print(sum(pca_obj.explained_variance_ratio_))

plt.figure(figsize = (8,8))

plt.plot(np.cumsum(pca_obj.explained_variance_ratio_))

plt.xlabel('n components')
plt.ylabel('Cumilative variance')
print(plt.show())

_, X_new = apply_pca(4)
print(X_new.sample(10))
Y = diabetis_data['Outcome']
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X_new,Y,test_size=0.30 )
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# Model accuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score
model_accuracy = accuracy_score(Y_test, Y_pred)
model_precision = precision_score(Y_test, Y_pred)
model_recall = recall_score(Y_test, Y_pred)
print("Accuracy of the model is {}% " .format(model_accuracy * 100))
print("Precision of the model is {}% " .format(model_precision * 100))
print("recall of the model is {}% " .format(model_recall * 100))







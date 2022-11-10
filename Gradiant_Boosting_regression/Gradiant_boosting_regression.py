# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:48:20 2022

@author: Prasad
"""

import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
insur_data= pd.read_csv('insurance_processed.csv')

print(insur_data.shape)

X = insur_data.drop('charges', axis=1)
Y = insur_data['charges']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=0.2)
print(" Train Data:  ",X_train.shape,  Y_train.shape )
print(" Test Data  ",X_test.shape, "   ", Y_test.shape )

from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=3)
tree_reg1.fit(X_train,Y_train)

y2 = Y_train - tree_reg1.predict(X_train)

tree_reg2= DecisionTreeRegressor(max_depth=3)
tree_reg2.fit(X_train,y2)

y3 = y2 - tree_reg2.predict(X_train) 
tree_reg3= DecisionTreeRegressor(max_depth=3)
tree_reg3.fit(X_train, y3)

y_pred = sum(tree.predict(X_test) for tree in (tree_reg1, tree_reg2, tree_reg3))
from sklearn.metrics import r2_score

print("r2_score without GradiantBoost but manual steps:", r2_score(Y_test,y_pred))

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(max_depth=3, n_estimators=3,learning_rate=1.0)
gbr.fit(X_train, Y_train)
y_gbr_pred=gbr.predict(X_test)
print("r2_score with GradiantBoost:", r2_score(Y_test,y_gbr_pred))





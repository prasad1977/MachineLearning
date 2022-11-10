import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

print(sklearn.__version__)

adv_data= pd.read_csv('Advertising.csv', index_col=0)

print(adv_data.head())
print(adv_data.shape)
print(adv_data.describe())

plt.Figure(figsize=(8,8))
plt.scatter(adv_data['Newspaper'],adv_data['Sales'],c='y')
print(plt.show())
plt.scatter(adv_data['Radio'],adv_data['Sales'],c='y')
print(plt.show())
plt.scatter(adv_data['TV'],adv_data['Sales'],c='y')
print(plt.show())

adv_data_corrrelation = adv_data.corr()
print(adv_data_corrrelation)

fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(adv_data_corrrelation,annot=True)

#model preparion

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = adv_data.drop('Sales', axis=1)
Y = adv_data['Sales']
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.30, random_state=0 )

import statsmodels.api as sm
from sklearn.metrics import r2_score


X_train_with_intercept = sm.add_constant(X_train)
stats_model = sm.OLS(Y_train,X_train_with_intercept )
fit_model = stats_model.fit()
print(fit_model.summary())
print('Model implementation started')

lm = LinearRegression()
model = lm.fit(X_train, Y_train)
print("Model Score on train data", model.score(X_train, Y_train))

"""
for i in range(len(X_test)):
    print( "X_test ",   X_test[i], " |y_pred" , Y_pred[i])
"""
Y_pred = model.predict(X_test)
print("Testing R2 Score : ", r2_score(Y_test, Y_pred))

def adjusted_r2 (r_square, labels, features):
    adj_r_square = 1 - ((1-r_square) * (len(labels) -1 )) / ((len(labels) - features.shape[1]))
    return adj_r_square

print("Adjusted R2 Score : ", adjusted_r2(r2_score(Y_test, Y_pred), Y_test, X_test))


plt.figure(figsize=(8,8))
plt.scatter(X_test, Y_test, c='black')

plt.plot(X_test,Y_pred,c='blue',linewidth=2)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()





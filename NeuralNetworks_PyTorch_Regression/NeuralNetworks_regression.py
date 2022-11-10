# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:29:39 2022

@author: Prasad
"""

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

from sklearn import preprocessing
adv_data[['TV']] = preprocessing.scale(adv_data[['TV']])
adv_data[['Radio']] = preprocessing.scale(adv_data[['Radio']])
adv_data[['Newspaper']] = preprocessing.scale(adv_data[['Newspaper']])
print(adv_data.sample(10))

X = adv_data.drop('Sales', axis=1)
Y = adv_data[['Sales']]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
print(X_train.shape," ",Y_train.shape)
import torch
x_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
x_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float)
y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float)


print(x_train_tensor.shape," ",y_train_tensor.shape)
print(x_test_tensor.shape," ",y_test_tensor.shape)



inp = 3
out=1

hid=100

loss_fn = torch.nn.MSELoss()
learning_rate = 0.0001

model = torch.nn.Sequential(torch.nn.Linear(inp,hid),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hid,out))

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(10000):
    y_pred = model(x_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    if iter % 1000 == 0 :
        print(iter, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
y_pred_tensor = model(x_test_tensor)
print(y_pred_tensor[:5])
    
y_pred = y_pred_tensor.detach().numpy()

plt.figure(figsize=(8,8))
plt.scatter(y_pred, Y_test.values)
plt.xlabel("Actual Sale")
plt.ylabel("Predicted Sale")
plt.title("Predicted Sale Vs Actual Sale")
plt.show()

from sklearn.metrics import r2_score
print("Testing R2 Score : ", r2_score(Y_test, y_pred))























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

drivers_data= pd.read_csv('driver-data.csv')

print(drivers_data.head(10))
print(drivers_data.shape)
print(drivers_data.describe())

drivers_data = drivers_data.sample(frac=1)

drivers_data.drop("id", axis=1, inplace=True)
print(drivers_data.sample(10))

fig , ax = plt.subplots(figsize=(10,8))
plt.scatter(drivers_data['Distance_Feature'],drivers_data['Speeding_Feature'],
            s=300, c='blue')
plt.xlabel('Distance_Feature')
plt.ylabel('Speeding_Feature')

print(plt.show())


from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=2,max_iter=1000)
kmeans_model.fit(drivers_data)
print(kmeans_model.labels_[::40])
print("Cluster Lables: ", np.unique ( kmeans_model.labels_))

zipped_list = list(zip(np.array(drivers_data),kmeans_model.labels_ ))

print("label and data: \n", zipped_list[1000:1000])

centeriods = kmeans_model.cluster_centers_

print("centeriods:", centeriods)

"""
colors = ['g' , 'y', 'b', 'k']
plt.figure(figsize=(10,8))

for i in zipped_list :
    plt.scatter(i[0][0],i[0][1],c=colors[ (i[1] % len(colors))])

plt.scatter(centeriods[:,0], centeriods[:,1], c='r', s=200, marker='s')

for j in range(len(centeriods)):
    plt.annotate(j, (centeriods[j][0], centeriods[j][1] ), fontsize=20)
    
print(plt.show()) """
    
from sklearn.metrics import silhouette_score       
    
print("Silhouette score ",silhouette_score(drivers_data,kmeans_model.labels_ ) )  
    
    
    
    
    
    
    
    

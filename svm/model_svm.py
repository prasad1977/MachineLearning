import numpy as np
import pandas as pd
filename='ad.data'
df=pd.read_csv(filename,sep=",",header=None,low_memory=(False))
print(df.head(10))

def toNum(cell):
    try:
        return np.float64(cell)
    except:
        return np.nan

def seriestoNum(series):
    return series.apply(toNum)

"""train_data=pd.isnull(df.iloc[0:,0:-1])
print(train_data.head(12))
"""
train_data=df.iloc[0:,0:-1].apply(seriestoNum)
print(train_data.head(12))
train_data=train_data.dropna()
print(train_data.head(12))

def toLabel(str):
    if str=="ad.":
        return 1
    else:
        return 0

train_labels=df.iloc[train_data.index,-1].apply(toLabel)

from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit=(train_data[100:2300],train_labels[100:2300])

#clf.predict(train_data.iloc[12].values)
temp=clf.predict(train_data.iloc[12].reshape(1,-1))
print(temp)

















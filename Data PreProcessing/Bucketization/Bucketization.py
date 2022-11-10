# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:14:39 2022

@author: Prasad
"""

import numpy as np
import pandas as pd

X = np.array([-7, 2, -3, -11, 14, 6, 8])

categories = pd.cut(X, 4)

print(categories.categories)
print(categories.codes)

marks=np.array([70, 20, 30, 99, 40, 16, 80])

categories, bins= pd.cut(marks, 4, retbins=True, labels=['poor', 'average', 'good', 'excellent'])
print(categories)


from sklearn.preprocessing import KBinsDiscretizer
encoder = KBinsDiscretizer(n_bins=4, encode = 'ordinal', strategy='uniform')
encoder.fit(marks)
encoder.transform(marks)

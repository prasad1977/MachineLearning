import pandas as pd
import numpy as np
import scipy.sparse as sparse
import random
import implicit 

datafile = 'u.data'
data = pd.read_csv(datafile, sep="\t", header=None, usecols=[0,1,2], names=['userId','itemId','rating'])
print(data.head())

data['userId']= data['userId'].astype("category")
data['itemId']= data['itemId'].astype("category")

"""
rating_matrix = coo_matrix((data['rating'].astype(float),
                            (data['itemId'].cat.codes.copy(),
                             data['userId'].cat.codes.copy())))
"""
sparse_item_user = sparse.csr_matrix((data['rating'].astype(float), (data['itemId'], data['userId'])))
sparse_user_item = sparse.csr_matrix((data['rating'].astype(float), (data['userId'], data['itemId'])))
print('After')

#user_factors, item_factors = implicit.alternating_least_squares(rating_matrix,factors=10,regularization=0.01)
model= implicit.als.AlternatingLeastSquares(factors=10,regularization=0.01,iterations=10)
alpha_val = 40
data_conf = (sparse_item_user * alpha_val).astype('double')
model.fit(data_conf)
print('After')

#user196 = item_factors.dot(user_factors[196])
user_id =   196
recommended = model.recommend(user_id,sparse_user_item)
print(recommended)


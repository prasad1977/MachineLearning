# Naive Bayes
import numpy
import pandas
import re

filename='forestfires.csv'

with open ("imdb_labelled.txt","r") as text_file:
    lines= text_file.read().split('\n')

with open ("amazon_cells_labelled.txt","r") as text_file:
    lines += text_file.read().split('\n')

with open ("yelp_labelled.txt","r") as text_file:
    lines += text_file.read().split('\n')

lines = [line.split("\t") 
           for line in lines
               if len(line.split("\t"))==2 and line.split("\t")[1] != '']         

train_documents=[line[0] for line in lines]
#print(train_documents)
train_labels=[line[1] for line in lines]
#print(train_labels)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
count_vectorizer= CountVectorizer(binary = 'true')
                                
train_documents=count_vectorizer.fit_transform(train_documents)
#print(train_documents[0])

classifier= BernoulliNB().fit(train_documents,train_labels)

out=classifier.predict(count_vectorizer.transform(['This is the best movie']))
print(out)
out=classifier.predict(count_vectorizer.transform(['This is the worst movie']))
print(out)

"""
strs = "foo\tbar\t\tspam"
re.split(r'\t+',strs)"""
"""
for strs in lines:
        print(strs)

    
    if re.split(r'\t+',strs).__sizeof__ == 2 :
        print(strs)"""



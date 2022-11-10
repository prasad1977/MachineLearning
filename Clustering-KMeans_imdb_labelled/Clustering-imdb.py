import csv

with open ("imdb_labelled.txt","r") as text_file:
    lines= text_file.read().split('\n')
    
lines = [line.split("\t") 
              for line in lines
                  if len(line.split("\t"))==2 and line.split("\t")[1] != '']

train_documents = [line[0] for line in lines]

from sklearn.feature_extraction.text import TfidfVectorizer

tfidfVectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
train_documents = tfidfVectorizer.fit_transform(train_documents)

from sklearn.cluster import KMeans

km= KMeans(n_clusters=3, init = 'k-means++' , max_iter = 100, n_init = 1, verbose=True)
km.fit(train_documents)

count=0
for i in range(len(lines)):
    if count > 3:
        break
    if km.labels_[i] == 2:
        print(lines[i])
        count+=1
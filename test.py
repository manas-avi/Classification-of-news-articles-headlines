import re
import string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


##################################
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
#################################	used for pca method of feature selection

from nltk.stem.porter import * # to perform stemming use to remove verbs ------

from nltk.corpus import * # for removing stop words

from collections import Counter
# to count the number of uinque words

# to convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import CountVectorizer
# for encoding categories

from sklearn.feature_selection import mutual_info_classif
# for feature selection

from sklearn.feature_selection import VarianceThreshold
# for feature selection

from sklearn.metrics import confusion_matrix
# for implementing confusion matrix

# from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# implements what is called one-of-K or “one-hot” coding for categorical (aka nominal, discrete) features. 
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest, chi2
# for feature selection by chisquare

# splitting the data in test and train
from sklearn.model_selection import train_test_split

# classifying with naive bayes
from sklearn.naive_bayes import MultinomialNB

#for classification
from sklearn import svm

# to check score
from sklearn.metrics import accuracy_score

# Neural network for calssification
from sklearn.neural_network import MLPClassifier

from sklearn import decomposition
# pca decompostion for feature selection

import matplotlib.pyplot as plt
import itertools
# these two arae for graphs

from sklearn.externals import joblib
# for saving model



# test_news = ["sensex rises above 30000 again","Movie Name: Baahubali 2: The Conclusion","Samsung Galaxy J3 Prime Budget Smartphone With Android 7.0 Nougat Launched","Drinking Coffee Can Reduce Prostate Cancer Risk","cancer","movie","isro launches 108 satellites into space","science","iitb award science"]
test_news=["Big rate changes unlikely under GST: Finance minister Arun Jaitley",
"Industry not doing its bit in creating jobs: NITI Aayog Vice-Chairman Arvind Panagariya",
"Tata-DoCoMo dispute: Delhi High Court approves $1.18 billion settlement",
"7th Pay Commission: Lavasa panel suggests changes in allowances for government employees",
"GST to push India GDP growth rate above 8 but bad loans a concern: IMF",
"Movie Name: Baahubali 2: The Conclusion",
"State report predicts sea levels rising due to polar ice melting",
"AIDS control programme running blind without enough testing kits",
"Beware! Diet Food Products Can Make You Gain Weight",
"Drinking Coffee Can Reduce Prostate Cancer Risk",
"Damning testimonies of the women scarred by rogue surgeon",
"Rising levels of carbon dioxide may change crucial marine process",
"To colonize space, start closer to Earth",
"Time travel is 'possible' -- mathematically anyway",
"NASA is running out of spacesuits and it could jeopardize future missions"]

file = open("features.txt", "r") 
features=file.read()
file.close()
features_list=features.split()
stop = set(stopwords.words('english'))
filtered_words = []
for sentence in test_news:
	filtered_words.append(" ".join([word.lower() for word in sentence.split() if (word.lower() not in stop)])) #need to check which words are removed
stemmer = PorterStemmer()
stemmed=[]
for words in filtered_words:
	st=''
	for i in words.split(' '):
		if stemmer.stem(i) in features_list:
			st=st+' '+stemmer.stem(i)

	stemmed.append(st)


stemmed.append(features)
vector = TfidfVectorizer()
x = vector.fit_transform(stemmed)

nb = joblib.load('nb.pkl') 

ypred = nb.predict(x)

for i in range(len(ypred)-1):
	prediction=''
	if ypred[i]== 0:
		prediction='business'
	elif ypred[i]== 3:
		prediction='science and technology'
	elif ypred[i]== 1:
		prediction='entertainment'
	elif ypred[i]== 2:
		prediction='health'
	print(test_news[i],"  -  ",prediction)

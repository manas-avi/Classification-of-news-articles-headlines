import re
import os
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

# ----------------------- Imported all the libraries ----------------------------------------------
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# -----------------------  function for plotting ----------------------------------------------------------

# grabbing the data
news_headlines = pd.read_csv("uci-news-aggregator.csv")
news=news_headlines.head(n=100000)
news_list = [s.lower().translate(str.maketrans('','',string.punctuation)) for s in news['TITLE']]

stop = set(stopwords.words('english'))
filtered_words = []
for sentence in news_list:
	filtered_words.append(" ".join([word for word in sentence.split() if ((word.lower() not in stop) and (len(word.lower())<15) and not (is_number(word.lower()))) ])) #need to check which words are removed

stemmer = PorterStemmer()
stemmed=[]
for words in filtered_words:
	st=''
	for i in words.split(' '):
		st=st+' '+stemmer.stem(i)

	stemmed.append(st)


# --------------------------------------------------------------------- data processing over ---------------------------------------------------------------------

# unique_words = list(set(" ".join(stemmed).split(" ")))
# size = len(unique_words)
# print(size) # to check the dimensionality of data


# ---------- Printing unique words --------------------------------------

# vector = TfidfVectorizer()
vector = CountVectorizer()
x = vector.fit_transform(stemmed)

code = LabelEncoder()
y = code.fit_transform(news['CATEGORY'])	# it takes the whole coloumn and then generates the required amount of labels

class_names = list(set(" ".join(news['CATEGORY']).split(" ")))


# ------------------------------------------------- feature selection ----------------------------------------------------------------

# uncomment them to implemet currently no feature reduction technique is implemented as time requires large amount of processing time

# mi = mutual_info_classif(x, y, discrete_features=True)
# x_chi = SelectKBest(chi2, k=1000).fit_transform(x, y)
# x_MI = SelectKBest(mutual_info_classif, k=10000).fit_transform(x, y)
# x_vt = VarianceThreshold().fit_transform(x) # it is set to its default value which is 
# pca = decomposition.TruncatedSVD(n_components=10) # pca.fit(x) # x_pca = pca.transform(x) # depriciated implementation of pca

# svd = decomposition.TruncatedSVD(n_components=10000)
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(svd, normalizer)
# x_pca = lsa.fit_transform(x)
# min_max_scaler = MinMaxScaler()
# x_pca = min_max_scaler.fit_transform(x_pca)


# ---------------split into train and test sets -----------------------------

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
# xtrain1, xtest1, ytrain1, ytest1 = train_test_split(x_chi, y, test_size=0.2)
# xtrain2, xtest2, ytrain2, ytest2 = train_test_split(x_MI, y, test_size=0.2)
# xtrain3, xtest3, ytrain3, ytest3 = train_test_split(x_vt, y, test_size=0.2)
# xtrain4, xtest4, ytrain4, ytest4 = train_test_split(x_pca, y, test_size=0.2)

# ------------------------------------------------naive bayes implementation ------------------------------------------------



featurenames=' '.join(vector.get_feature_names())
file = open("features.txt","w") 
file.write(featurenames)
file.close()
# print(xtrain4)
nb = MultinomialNB()
nb.fit(xtrain, ytrain)	# fitting the text
print("Normal implementation")
print(nb.score(xtest, ytest))		# to calculate the score for the classification
joblib.dump(nb, 'nb.pkl') 

# ------------------------------------------------sended this features------------------------------------------------

# ucmment others to use them

# nb = MultinomialNB()
# nb.fit(xtrain4, ytrain4)	# fitting the text
# print("With pca")
# print(nb.score(xtest4, ytest4))		# to calculate the score for the classification

# clf = RandomForestClassifier(n_estimators=1000)
# clf = clf.fit(xtrain4,ytrain4)
# print("With Random forest cllassifier on pca")
# print(clf.score(xtest4, ytest4))


# nb = MultinomialNB()
# nb.fit(xtrain1, ytrain1)	# fitting the text
# print("with chi square ")
# print(nb.score(xtest1, ytest1))		# to calculate the score for the classification

# nb = MultinomialNB()
# nb.fit(xtrain2, ytrain2)	# fitting the text
# print("with Mi ")
# print(nb.score(xtest2, ytest2))		# to calculate the score for the classification


# nb = MultinomialNB()
# nb.fit(xtrain3, ytrain3)	# fitting the text
# print("with Variation thrshold ")
# print(nb.score(xtest3, ytest3))		# to calculate the score for the classification



# --------------------------- uncomment this to implemet various svm classifiers --------------------------------------

# C = 1.0 # parameters
# print("linear kernel without MI")
# svc = svm.SVC(kernel='linear', C=1.0).fit(xtrain,ytrain)
# print(accuracy_score(ytest, svc.predict(xtest)))
# print("linear kernel with MI")
# svc = svm.SVC(kernel='linear', C=1.0).fit(xtrain2,ytrain2)
# print(accuracy_score(ytest2, svc.predict(xtest2)))
# # rbfsvc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(xtrain,ytrain)
# # print(accuracy_score(ytest, rbfsvc.predict(xtest)))
# # polysvc = svm.SVC(kernel='poly', degree=3, C=C).fit(xtrain,ytrain)
# # print(accuracy_score(ytest, polysvc.predict(xtest)))

# # both of their efficiency is not good

# linsvc = svm.LinearSVC(C=C).fit(xtrain,ytrain)
# print("linear SVC without MI")
# print(accuracy_score(ytest, linsvc.predict(xtest)))

# linsvc = svm.LinearSVC(C=C).fit(xtrain2,ytrain2)
# print("linear SVC with MI")
# print(accuracy_score(ytest2, linsvc.predict(xtest2)))


# ------------------------------------------ uncomment this to implement neural Network----------------------------------------------------

# NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 2), random_state=1).fit(xtrain,ytrain)
# print("NN without MI")
# print(accuracy_score(ytest, NN.predict(xtest)))

# NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 2), random_state=1).fit(xtrain2,ytrain2)
# print("NN with MI")
# print(accuracy_score(ytest2, NN.predict(xtest2)))

# ---------------------- Plotting ------------------------------------

ypred = nb.predict(xtest)
cnf_matrix = confusion_matrix(ytest, ypred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

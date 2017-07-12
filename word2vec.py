from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import * # to perform stemming use to remove verbs ------
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import string

from nltk.corpus import * # for removing stop words


import pandas as pd
import numpy as np

news_headlines = pd.read_csv("uci-news-aggregator.csv")
news=news_headlines.head(n=10000)
news_list = [s.lower().translate(str.maketrans('','',string.punctuation)) for s in news['TITLE']]
# print(len(news_list))
# news_list=pd.Series.tolist(lines)
stop = set(stopwords.words('english'))
filtered_words = []
for sentence in news_list:
	filtered_words.append(" ".join([word for word in sentence.split() if word.lower() not in stop])) #need to check which words are removed


stemmer = PorterStemmer()
stemmed=[]
for words in filtered_words:
	st=''
	for i in words.split(' '):
		st=st+' '+stemmer.stem(i)

	stemmed.append(st)
code = LabelEncoder()
y = code.fit_transform(news['CATEGORY'])

X = []

for line in stemmed:
        # label, text = line.split(" ")
        # texts are already tokenized, just split on space
        # in a real case we would use e.g. spaCy for tokenization
        # and maybe remove stopwords etc.
        X.append(line.split())
        # y.append(label)

X, y = np.array(X), np.array(y)

with open('glove.6B.300d.txt',"rb") as lines:
	word2vec = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}

model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
model.wv.index2word
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        # print(word2vec.values())
        self.dim = len(next(iter(word2vec.values())))
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])

print(cross_val_score(etree_w2v_tfidf, X, y).mean())

# xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
# print(print(etree_w2v_tfidf.score(xtest, ytest)))
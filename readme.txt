train.py -
contains the code for training the data for news-aggregator dataset.
Only naive bayes will be  currently implemented if ran directly. Others functionalities are commented out and can be uncommented to implement them.
Execution command - python3 train.py

test.py -
contains the code for checking the implementation of our trained data set.
Append the news one want to classify in the message
Execution command - python3 test.py


word2vec.py -
this includes the implementation of the word2vec method (it uses two set of corpus glove.6B.50d.txt and glove.6B.300d.txt (smaller corpus set) (bigger corpus set) which needs to be downloaded from the following site)
website - https://nlp.stanford.edu/projects/glove/
citation - https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking.ipynb (some part of the code was taken from this website so we want to acknowledge them)
Execution command - python3 word2vec.py


feature.txt - helper file to store the features

nb.pkl - contains the trained model

Link for dataset "uci-news-aggregator.csv" -  https://www.kaggle.com/uciml/news-aggregator-dataset/downloads/news-aggregator-dataset.zip
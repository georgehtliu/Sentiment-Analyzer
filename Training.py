#Import Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import f1_score
# from sklearn.svm import SVC
# from sklearn.linear_model import SGDClassifier
import nltk
import string
import re

df = pd.read_csv('training_data.csv')
df = df[['Text','Sentiment']]

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df['Text'] = df['Text'].map(lambda text: remove_punct(text))
df.head(20) #Examine the data

X = np.array(df['Text'])
y = np.array(df['Sentiment'])

vectorizer = TfidfVectorizer(strip_accents='unicode', max_df=0.5, ngram_range=(1,2))
X_train_vectors = vectorizer.fit_transform(X)

from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression(solver='sag', max_iter=10000, C=2, class_weight='dict')
clf_log.fit(X_train_vectors, y)

import pickle
pickle_out = open('Vectorizer','wb')
Pickle = pickle.dump(vectorizer, pickle_out)
pickle_out.close()

pickle_out = open('SentimentNewton-Log','wb')
Pickle = pickle.dump(clf_log, pickle_out)
pickle_out.close()
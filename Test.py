import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
## Data is split 50-50 between 0-1
df = pd.read_csv('training_data.csv')
df = df[['Text','Sentiment']]


mini_df = df.sample(2000)

X = np.array(mini_df['Text'])
y = np.array(mini_df['Sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

## Tfidf works better than count vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

parameters = {'kernel': ('linear', 'rbf'), 'C': (1,4,8,16,32)}
svc = SVC()
clf = GridSearchCV(svc, parameters, cv = 5)

clf.fit(X_train_vectors, y_train)
print(clf)

## Around 68% accuracy using 8000 of the 1M training examples
print(f1_score(y_test, clf.predict(X_test_vectors), average=None, labels=[0,1]))

print(clf.get_params())
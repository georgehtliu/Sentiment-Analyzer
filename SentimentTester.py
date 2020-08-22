import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

## Data is split 50-50 between 0-1
df = pd.read_csv('training_data.csv')
df = df[['Text','Sentiment']]

print(df['Sentiment'].value_counts())

mini_df = df[:8000]

X = np.array(mini_df['Text'])
y = np.array(mini_df['Sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

clf = MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes=(64,64))
clf.fit(X_train_vectors, y_train)


## Around 68% accuracy using 8000 of the 1M training examples
print(f1_score(y_test, clf.predict(X_test_vectors), average=None, labels=[0,1]))




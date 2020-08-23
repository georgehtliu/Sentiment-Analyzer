import joblib
import pandas as pd

filename = 'SentimentNewton-Log'
clf_log = joblib.load(filename)

filename = 'Vectorizer'
vectorizer = joblib.load(filename)

df = pd.read_csv('contestant_judgment.csv')
df['Sentiment'] = df['Text'].apply(lambda x: clf_log.predict(vectorizer.transform([x]))[0])
df.to_csv('JudgingResults.csv', index=None)
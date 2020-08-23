import joblib
import pandas as pd
import csv


filename = 'SentimentNewton-Log'
clf_log = joblib.load(filename)

filename = 'Vectorizer'
vectorizer = joblib.load(filename)

df = pd.read_csv('contestant_judgment.csv')
df['Sentiment'] = df['Text'].apply(lambda x: clf_log.predict(vectorizer.transform([x]))[0])

with open('JudgingResults.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['ID','User','Text','Sentiment'])
    for i,_ in enumerate(df['Text']):
        writer.writerow([str(df.iloc[i]['ID']), str(df.iloc[i]['User']), str(df.iloc[i]['Text']),
                         str(df.iloc[i]['Sentiment'])])


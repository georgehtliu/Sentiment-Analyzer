import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('training_data.csv')

df = df[['Text','Sentiment']]

print(df.head())

new_df = df[:1000]
print(new_df.size)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('training_data.csv')

df = df[['Text','Sentiment']]

print(df.head())

new_df = df[:1000]
print(new_df.size)

X = new_df['Text']
y = new_df['Sentiment']


import pandas as pd

df = pd.read_csv('training_data.csv')

df = df[['Text','Sentiment']]

print(df.head())
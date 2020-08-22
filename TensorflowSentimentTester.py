import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('training_data.csv')

df = df[['Text','Sentiment']]

mini_df = df.sample(5000)

X = np.array(mini_df['Text'])
y = np.array(mini_df['Sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = tf.keras.Sequential()


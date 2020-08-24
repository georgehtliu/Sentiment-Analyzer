# Sentiment Analyzer
Submission for the 2020 Ignition Hacks Sigma Division. We created an AI model that predicts the sentiment of a given sentence, classifying it as positive (represented with a 1) or negative (represented with a 0). The code was written in Python using the scikit-learn machine learning library and the Natural Language Toolkit.
### Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites
To run on a local device, you will need to download the following:\
- Pandas\
- Numpy\
- Sklearn\
- NLTK\
Alternatively, you may run on a cloud-hosted development environment.
### Running the program
1. Running the “submission_training.ipynb” file will save the logistic regression model under the file name “SentimentNewton_Log.pkl” and the TfidfVectorizer under the file name “Vectorizer.pkl “.
2. From there, running “submission_createcsv.ipynd” will use the regression model to predict the sentiments of the judgment dataset 
### Data Cleaning
- Removed punctuation utilizing regex string query function\
- Used lemmatization functions and removed names\
- Used a Tfidfvectorizor to further clean
### Optimization
- Considered many different models including a Neural Network and support vector machines\
- Other classifiers and natural language processing operations considered in the submission_extras.ipynb file\
- Utilized the Logistic regression model\
- Optimized the parameters of the logistic regression using GridSearchCV\
- Optimizing the vectorizer by trying different techniques to clean data, such as using lemmatization and stopwords\
- Extracted bigrams along with unigrams in the vectorizer\
- Obtained an f1 score of around [0.82,0.82]\

### Version History
Version 1.0: This is the first release of the Sentiment Analyzer
### Authors
- David Chen\
- George Liu\
- David Wang\
- Michael Yang
### License
This project is licensed under the MIT License - see the License file in the git repository for details
### Acknowledgments
The scikit-learn library was used extensively throughout the project


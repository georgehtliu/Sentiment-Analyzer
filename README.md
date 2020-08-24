# Sentiment Analyzer
Submission for the 2020 Ignition Hacks Sigma Division. We created an AI model that predicts the sentiment of a given sentence, classifying it as positive (represented with a 1) or negative (represented with a 0). The code was written in Python using the scikit-learn machine learning library and the Natural Language Toolkit.
### Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites
To run on a local device, you will need to download the following:\
Pandas\
Numpy\
Sklearn\
NLTK
Alternatively, you may run on a cloud-hosted development environment.
### Running the program
Explain how to run the automated tests for this system
### Data Cleaning
Removed punctuation utilizing regex string query function\
Used a Tfidfvectorizor to further clean
### Optimization
Logistic regression model\
Optimized the parameters of the linear regression using a grid search\
Optimizing the vectorizor by trying different techniques to clean data, such as using lemmatization and stopwords\
Obtained an F1 score of [0.82,0.82]\
### Version History
Version 1.0: This is the first and last release of the Sentiment Analyzer
### Authors
David Chen\
George Liu\
David Wang\
Michael Yang
### License
This project is licensed under the MIT License - see the LICENSE.md file for details
### Acknowledgments
The scikit-learn library was used extensively throughout the project


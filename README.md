# Sentiment Analyzer 🎯

A machine learning-powered sentiment analysis tool that classifies text as positive or negative sentiment. This project won **Second Place for accuracy** at the 2020 Ignition Hacks Sigma Division.

## Overview

The Sentiment Analyzer uses a logistic regression model with TF-IDF vectorization to predict sentiment polarity (positive = 1, negative = 0) in text data. The model achieved an F1 score of approximately 0.82.

**Project Link:** [Devpost Submission](https://devpost.com/software/sentiment-analyzer-jqhfda)

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data Cleaning](#data-cleaning)
- [Model Optimization](#model-optimization)
- [Performance](#performance)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- ✅ Binary sentiment classification (positive/negative)
- ✅ Preprocessing pipeline with punctuation removal
- ✅ TF-IDF vectorization with bigrams and unigrams
- ✅ Optimized logistic regression model
- ✅ GridSearchCV for hyperparameter tuning
- ✅ F1 score of ~0.82

## Prerequisites

To run this project on your local machine, you'll need:

- **Python 3.6+**
- Required Python packages:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` (sklearn) - Machine learning library
  - `nltk` - Natural Language Toolkit
  - `joblib` - Model serialization

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/georgehtliu/sentiment-analyzer.git
   cd sentiment-analyzer
   ```

2. **Install required packages:**
   ```bash
   pip install pandas numpy scikit-learn nltk joblib
   ```

3. **Download NLTK data (if needed):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

Alternatively, you may run this project on a cloud-hosted development environment such as Google Colab.

## Project Structure

```
Sentiment-Analyzer/
├── README.md                      # This file
├── LICENSE.md                     # MIT License
├── training_data.csv              # Training dataset
├── contestant_judgment.csv        # Judgement/test dataset
├── predicted_labels.csv           # Output predictions
├── submission_training.ipynb      # Model training notebook
├── submission_createcsv.ipynb     # Prediction generation notebook
└── submission_extras.ipynb        # Additional experiments and analysis
```

**Note:** The trained model files (`SentimentNewton_Log.pkl` and `Vectorizer.pkl`) are generated when running the training notebook.

## Usage

### Step 1: Train the Model

Run the training notebook to train and save the sentiment analysis model:

```bash
jupyter notebook submission_training.ipynb
```

This notebook will:
- Load and preprocess the training data
- Train a logistic regression model with optimized hyperparameters
- Save the trained model as `SentimentNewton_Log.pkl`
- Save the TF-IDF vectorizer as `Vectorizer.pkl`

### Step 2: Generate Predictions

After training, use the prediction notebook to classify new text:

```bash
jupyter notebook submission_createcsv.ipynb
```

This notebook will:
- Load the trained model and vectorizer
- Process the judgement dataset (`contestant_judgment.csv`)
- Generate predictions and save them to `predicted_labels.csv`

**Note:** Make sure `contestant_judgment.csv` is in the same directory, or update the file path in the notebook.

## Data Cleaning

The preprocessing pipeline includes:

- **Punctuation removal** using regex patterns
- **TF-IDF vectorization** with the following features:
  - Removal of stopwords
  - Extraction of both unigrams and bigrams
  - Tested lemmatization (results documented in `submission_extras.ipynb`)

The team experimented with various cleaning techniques, and found that punctuation removal alone provided the best accuracy for this dataset.

## Model Optimization

The model was developed through extensive experimentation:

### Model Selection
- Tested multiple classifiers including:
  - Neural Networks (MLPClassifier)
  - Support Vector Machines (SVC)
  - Logistic Regression (selected)
  - SGD Classifier
- **Logistic Regression** was chosen for optimal performance

### Hyperparameter Tuning
- Used **GridSearchCV** to optimize logistic regression parameters
- Optimized TF-IDF vectorizer settings:
  - Tuned n-gram range (unigrams and bigrams)
  - Experimented with stopword removal
  - Tested different lemmatization approaches

### Additional Experiments
Detailed exploration of alternative approaches and results can be found in `submission_extras.ipynb`.

## Performance

- **F1 Score:** ~0.82
- **Task:** Binary sentiment classification
- **Model:** Optimized Logistic Regression with TF-IDF features

## Authors

- **David Chen**
- **George Liu**
- **David Wang**
- **Michael Yang**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- **scikit-learn** - Extensive use of machine learning tools and algorithms
- **NLTK** - Natural language processing capabilities
- **Ignition Hacks 2020** - Competition organizers and judges
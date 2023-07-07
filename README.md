# Spam Email Classifier

This project implements a machine learning model to classify emails as spam or non-spam. It utilizes the Naive Bayes algorithm for text classification and demonstrates the process of training the model, evaluating its performance, and making predictions.

## Project Overview

- `ExploreDataset.py`: A Python script that reviews and preprocesses the "emails.csv" dataset from [Kaggle](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv). The script performs data exploration, handles missing values, and scales the features using MinMaxScaler.

- `SpamEmailClassifier.py`: A Python script that trains a Multinomial Naive Bayes classifier on the preprocessed dataset. It splits the dataset into training and testing sets, trains the model, and evaluates the model's performance using classification metrics and a heatmap visualization.

## Prerequisites

- Python 3.9.13
- pandas
- scikit-learn
- matplotlib
- seaborn

## Results

The `SpamEmailClassifier.py` script outputs a classification report that includes precision, recall, F1-score, and support for both spam and non-spam classes. It also displays a heatmap visualization representing the accuracy of the model.

## License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/license/mit/) file for more details.

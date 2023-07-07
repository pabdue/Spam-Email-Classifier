# -----------------------------------------------------------------------------------------------
# Name: Pablo Duenas
# Date: July 5, 2023
# Dataset: emails.csv
# Source: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv
# Description: Preprocess the dataset, emails.csv. Train and test Naive Bayes model. 
#              Display heatmap that represents model's accuracy.
# -----------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load emails.csv dataset
df = pd.read_csv('emails.csv')

# Dropping unnecessary column
df = df.drop(['Email No.'], axis=1)

# Feature scaling
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Split into training and testing sets
X = df.drop(['Prediction'], axis=1)  # Features
y = df['Prediction']  # prediction column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes classifier
model = MultinomialNB()

# Train NB classifier
model.fit(X_train, y_train)

# Prediction
y_prediction = model.predict(X_test)

# Evaluation of model's performance
report = classification_report(y_test, y_prediction)
print('Report of model\'s performance: \n', report)

'''
Output:

Report of model's performance: 
               precision    recall  f1-score   support

         0.0       0.95      0.96      0.96       739
         1.0       0.91      0.88      0.89       296

    accuracy                           0.94      1035
   macro avg       0.93      0.92      0.92      1035
weighted avg       0.94      0.94      0.94      1035
'''

print('Overall accuracy: 94%\n')

# True positives, true negatives, false positives, false negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_prediction).ravel()

# Labels and confusion matrix values
labels = ['Non-spam', 'Spam']
cm = np.array([[tn, fp], [fn, tp]])

# Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

# Add labels, title, and axis ticks
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Accuracy Report')
plt.xticks(ticks=[0.5,1.5], labels=labels)
plt.yticks(ticks=[0.5,1.5], labels=labels)

# Display heatmap
plt.show()
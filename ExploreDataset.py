# -----------------------------------------------------------------------------------------------
# Name: Pablo Duenas
# Date: July 4, 2023
# Dataset: emails.csv
# Source: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv
# Description: Reviewing and preprocessing the dataset (emails.csv)
# -----------------------------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Exploring the contents of the emails.csv dataset
print('________________ Exploring the dataset ________________\n')

# Load emails.csv dataset
df = pd.read_csv('emails.csv')

# print first 5 rows of dataset
print('Preview of dataset:\n', df.head(), '\n')

# check shape of dataset
print('Dataset shape: ', df.shape, '\n')

# check data types of all columns 
print('Feature data types:\n', df.dtypes, '\n')

# Check for missing values
print('Missing values:\n', df.isnull().sum(),'\n')

# -----------------------------------------------------------------------------------------------
# preprocessing dataset
print('________________ preprocessing ________________\n')

# dropping unnecessary columns
df = df.drop(['Email No.'], axis=1)

# feature scaling
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Split into training and testing sets
X = df.drop(['Prediction'], axis=1)  # Features
y = df['Prediction']  # prediction column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocessed dataset
print('Preview of updated dataset:\n', df.head(), '\n')

# Print the updated shape of the dataset
print('Updated emails.csv shape: ', df.shape)

'''
Output:

________________ Exploring the dataset ________________

Preview of dataset:
   Email No.  the  to  ect  and  for  of    a  you  hou  in  ...  enhancements  connevey  jay  valued  lay  infrastructure  military  allowing  ff  dry  Prediction
0   Email 1    0   0    1    0    0   0    2    0    0   0  ...             0         0    0       0    0               0         0         0   0    0           0
1   Email 2    8  13   24    6    6   2  102    1   27  18  ...             0         0    0       0    0               0         0         0   1    0           0
2   Email 3    0   0    1    0    0   0    8    0    0   4  ...             0         0    0       0    0               0         0         0   0    0           0
3   Email 4    0   5   22    0    5   1   51    2   10   1  ...             0         0    0       0    0               0         0         0   0    0           0
4   Email 5    7   6   17    1    5   2   57    0    9   3  ...             0         0    0       0    0               0         0         0   1    0           0

[5 rows x 3002 columns] 

Dataset shape:  (5172, 3002) 

Feature data types:
 Email No.     object
the            int64
to             int64
ect            int64
and            int64
               ...  
military       int64
allowing       int64
ff             int64
dry            int64
Prediction     int64
Length: 3002, dtype: object 

Missing values:
 Email No.     0
the           0
to            0
ect           0
and           0
             ..
military      0
allowing      0
ff            0
dry           0
Prediction    0
Length: 3002, dtype: int64 

________________ preprocessing ________________

Preview of updated dataset:
         the        to       ect       and       for        of         a       you  ...  valued  lay  infrastructure  military  allowing        ff  dry  Prediction
0  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.001054  0.000000  ...     0.0  0.0             0.0       0.0       0.0  0.000000  0.0         0.0
1  0.038095  0.098485  0.067055  0.067416  0.127660  0.025974  0.053741  0.014286  ...     0.0  0.0             0.0       0.0       0.0  0.008772  0.0         0.0
2  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.004215  0.000000  ...     0.0  0.0             0.0       0.0       0.0  0.000000  0.0         0.0
3  0.000000  0.037879  0.061224  0.000000  0.106383  0.012987  0.026870  0.028571  ...     0.0  0.0             0.0       0.0       0.0  0.000000  0.0         0.0
4  0.033333  0.045455  0.046647  0.011236  0.106383  0.025974  0.030032  0.000000  ...     0.0  0.0             0.0       0.0       0.0  0.008772  0.0         0.0

[5 rows x 3001 columns] 

Updated emails.csv shape:  (5172, 3001)
'''

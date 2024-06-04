# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:00:23 2024
https://github.com/aiplanethub/Datasets/blob/master/HR_comma_sep.csv
githubusercontent.com/selva86/datasets/master/employee.csv


@author: chenj
"""
import os
os.chdir("C:\\Users\\chenj\\Desktop\\Jun\\code\\GPT")

# first 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load example data
url = "HR_comma_sep.csv"
df = pd.read_csv(url)

# Display first few rows
df.head()


# Renaming columns for easier access
df.columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'time_spent_company', 'work_accident', 'left', 'promotion_last_5years', 'department', 'salary']

# Convert categorical variables to numeric
df = pd.get_dummies(df, columns=['department', 'salary'])

# Splitting the data into features and target
X = df.drop('left', axis=1)
y = df['left']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Initializing and training the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# Making predictions
y_pred = clf.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)



# Displaying feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Printing the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {df.columns[indices[f]]} ({importances[indices[f]]})")




















# 2 nd example
# This example will use a dataset to predict whether an employee will achieve a high performance rating based on various factors.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load example data (You would replace this with your actual dataset)
url = "https://raw.githubusercontent.com/selva86/datasets/master/employee.csv"
df = pd.read_csv(url)

# Display first few rows
df.head()

# Renaming columns for easier access (if necessary)
df.columns = [col.lower().replace(" ", "_") for col in df.columns]

# Convert categorical variables to numeric
df = pd.get_dummies(df, columns=['department', 'education', 'recruitment_channel', 'gender'])

# Handling missing values
df.fillna(df.mean(), inplace=True)

# Splitting the data into features and target
X = df.drop('is_promoted', axis=1)  # Assuming 'is_promoted' is the target column indicating high performance
y = df['is_promoted']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Initializing and training the model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)




# Making predictions
y_pred = clf.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)



# Displaying feature importance for logistic regression
coefficients = clf.coef_[0]
features = X.columns

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

print(importance_df)



# derstand factors contributing to employee satisfaction
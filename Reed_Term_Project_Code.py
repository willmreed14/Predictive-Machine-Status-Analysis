#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAP 4773 - Data Science Analytics
Term Project
@author: willreed
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings

# Ignore specific warnings about class populations being too small for the number of splits
warnings.filterwarnings("ignore", message="The least populated class in y has only 1 members, which is less than n_splits=")

"""Task 1: Data Preprocessing"""

''' Import data '''

# Select only sensor data columns (columns 3-54)
columns_indices = range(2, 54) 

# Read only the specified columns and rows (for now)
skip = lambda x: x not in range(127900 - 1) and x != 0

# Reading the data
sensor_data = pd.read_csv('sensor.csv', usecols=columns_indices, skiprows=skip, nrows=210)

# Set the index to the row numbers you are interested in
sensor_data.index = range(127900, 127900 + 210)

# Print the first few rows to verify the import
print(sensor_data)

''' Handle NaN Values '''

# Check for NaN values in each column of the DataFrame
missing_values_count = sensor_data.isna().sum()

# Print the number of missing values for each sensor column
#print("Missing values in each column:")
#print(missing_values_count)

# Drop sensor 15 column since it has no valid values
sensor_data.drop('sensor_15', axis=1, inplace=True)

# Check for NaN values again to verify they have been handled.
missing_values_count = sensor_data.isna().sum()

# Print the number of missing values for each sensor column
#print("Missing values in each column:")
#print(missing_values_count)

''' Handle Outliers '''

# Calculate Z-scores of each column
z_scores = np.abs(stats.zscore(sensor_data))

# Remove rows containing outliers based on z-score
sensor_data = sensor_data[(z_scores < 3).all(axis=1)]

print(sensor_data)

''' Standardize Data '''

# Round all values in the DataFrame to 5 decimal places
sensor_data = sensor_data.round(5)

# Print the DataFrame to verify the change
print(sensor_data.head())

""" Task 2: Exploratory Data Analysis """

# Plotting histograms for all sensor columns
sensor_data.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms of Sensor Data')
plt.show()

# Import machine status, excluding rows that were removed when cleaning outliers.
machine_status = pd.read_csv('sensor.csv', usecols=[54], skiprows=lambda x: x not in list(sensor_data.index + 1), header=None)

# Rename the column for clarity
machine_status.columns = ['Machine_Status']

# Verify import
#print(machine_status)

# Encode Categorical Data
encoder = LabelEncoder()
machine_status['Encoded_Status'] = encoder.fit_transform(machine_status['Machine_Status'])

# Reset the index of both DataFrames to ensure alignment
sensor_data.reset_index(drop=True, inplace=True)
machine_status.reset_index(drop=True, inplace=True)

# Create a new DataFrame storing both numerical and machine_status data
combined_data = pd.concat([sensor_data, machine_status], axis=1)

# Verify combined DF
#print(combined_data)

# Check the mapping to understand what each encoded number represents
status_mapping = {index: label for index, label in enumerate(encoder.classes_)}
print("Status Mapping:", status_mapping)

""" Task 3: Feature Engineering """

# Select only numeric columns for correlation (excluding 'Machine_Status' string column)
numeric_columns = combined_data.select_dtypes(include=[np.number])

# Calculate correlations between sensor columns and the encoded status
correlations = numeric_columns.corr()['Encoded_Status'].drop('Encoded_Status')

# Print sorted correlations to identify significant relationships
print("Correlation Values\n", correlations.sort_values(ascending=False))

# Prepare the data
X = combined_data[[col for col in combined_data.columns if 'sensor' in col]]
y = combined_data['Encoded_Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
sensor_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Print the feature importances
print("Feature Importances\n", sensor_importance)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x=sensor_importance.values, y=sensor_importance.index)
plt.title('Feature Importance of Sensors')
plt.xlabel('Importance Score')
plt.ylabel('Sensor Names')
plt.show()

""" Task 4: Predictive Modeling """

# Define which columns contain numerical sensor data
sensor_columns = [col for col in combined_data.columns if 'sensor' in col]
X = combined_data[sensor_columns]
y = combined_data['Encoded_Status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models
logistic_model = LogisticRegression(max_iter=5000)
random_forest_model = RandomForestClassifier(n_estimators=100)
svm_model = SVC()

# Train the models
logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Make predictions
logistic_predictions = logistic_model.predict(X_test)
rf_predictions = random_forest_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)

def evaluate_model(name, predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro', zero_division=0)
    recall = recall_score(y_test, predictions, average='macro', zero_division=0)
    f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
    print(f"{name} performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

# Evaluate each model as before
evaluate_model('Logistic Regression', logistic_predictions, y_test)
evaluate_model('Random Forest', rf_predictions, y_test)
evaluate_model('SVM', svm_predictions, y_test)

""" Task 5: Model Evaluation and Selection """

# Setting up 10-fold stratified cross-validation
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Function to perform cross-validation
def evaluate_cross_validation(model, X, y, cv, scoring):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores.mean()

# Prepare the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC()
}

# Dictionary to store results
results = {name: {} for name in models.keys()}

# Metrics to evaluate
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Evaluate each model
for name, model in models.items():
    print(f"Evaluating {name}")
    for metric in metrics:
        score = evaluate_cross_validation(model, X, y, cv, scoring=metric)
        results[name][metric] = score
        print(f"{metric}: {score:.4f}")
    print("\n")

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results).T
print(results_df)






















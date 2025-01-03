#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:18:42 2024
CAP 4773 - Data Science Analytics
Term Project
File 1
@author: willreed
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

'''Task 1: Data Preprocessing '''

# Specify the range of columns to import (Python uses 0-based indexing, so adjust accordingly)
# Columns 3 through 54 in 1-based index are columns 2 through 53 in 0-based index
columns_indices = range(2, 54)  # This includes columns 3 to 54 in a 1-based index system

# Read only the specified columns and the first 20 rows
data = pd.read_csv('sensor.csv', usecols=columns_indices, nrows=20)

# Print the first few rows to verify the import
print(data.head())

# Calculate z-scores
z_scores = np.abs(stats.zscore(data[sensor_columns]))

# Create a mask where all z-scores are less than 3, but apply it per column
filtered_entries = (z_scores < 3).all(axis=1)

# Apply the mask to the data
data_filtered = data[filtered_entries]

# Handle data noise using a rolling mean with a window size = 3
'''
for i in range(0, 51):
    if 10 > i >= 0:
        data[f'sensor_0{i}'] = data[f'sensor_0{i}'].rolling(window=3).mean()
    else:
        data[f'sensor_{i}'] = data[f'sensor_{i}'].rolling(window=3).mean()
'''       
 
# Forward Filling NaN values
data.ffill(inplace=True)

# Display the first few rows of the dataset to verify changes
print(data.head())

# Displaying summary statistics for the dataset
summary_stats = data.describe()
print(summary_stats)

# Selecting a subset of sensor columns for visualization
sensor_columns = [col for col in data.columns if 'sensor' in col]

# Plotting histograms for selected sensors
data[sensor_columns[:10]].hist(bins=50, figsize=(20, 15))
plt.show()

# Creating box plots for the first few sensors to check for outliers
data[sensor_columns[:10]].plot(kind='box', figsize=(20, 10), vert=False)
plt.show()

# Checking for columns with constant values
constant_columns = [col for col in sensor_columns if data[col].nunique() == 1]
print("Constant columns:", constant_columns)

# Checking data types
data_types = data.dtypes
non_numeric_columns = data_types[~data_types.isin(['int64', 'float64'])].index.tolist()
print("Non-numeric columns:", non_numeric_columns)

# Check for NaNs and Inf values
print("NaNs in each column:\n", data[sensor_columns].isna().sum())
#print("Infs in each column:\n", np.isinf(data[sensor_columns]).sum())

print(data.head())

# Check for NaN values and fill or drop them
if data[sensor_columns].isna().any().any():
    data[sensor_columns].fillna(data[sensor_columns].mean(), inplace=True)  # Fill NaNs with mean or choose another appropriate method

# Now apply the scaling
scaler = StandardScaler()
data[sensor_columns] = scaler.fit_transform(data[sensor_columns])








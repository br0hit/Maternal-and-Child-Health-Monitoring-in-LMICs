import pandas as pd
import random

# Loading datasets 
file_path = '../data/gee_features.csv'

# Specify the number of rows to skip
sample_size = 5000  # Desired sample size
# Get the total number of rows in the file
with open(file_path, 'r') as f:
    total_rows = sum(1 for _ in f) - 1  # Subtract 1 to exclude the header row

skip_rows = sorted(random.sample(range(1, total_rows + 1), total_rows - sample_size))

# gee_features = pd.read_csv('../data/gee_features.csv')



# Read the dataset with random sampling
training_labels = pd.read_csv('../data/training_label.csv')
# sample_submission = pd.read_csv('../data/sample_submission.csv')

# # Check missing values in training labels dataset
# missing_values = training_labels.isnull().sum()
# print(missing_values)
# print("\n\n\n")

# # Check missing values in gee features dataset
# missing_values = gee_features.isnull().sum()
# print(missing_values)

# print(gee_features["DHSID"].nunique())

import matplotlib.pyplot as plt

# Scatter plot of latitude and longitude
plt.scatter(training_labels['LONGNUM'], training_labels['LATNUM'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical Distribution')
plt.show()







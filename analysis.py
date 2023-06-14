import pandas as pd

# Loading datasets 
gee_features = pd.read_csv('../data/gee_features.csv', nrows=100)
training_labels = pd.read_csv('../data/training_label.csv')
# sample_submission = pd.read_csv('../data/sample_submission.csv')

# Check missing values in training labels dataset
missing_values = training_labels.isnull().sum()
print(missing_values)
print("\n\n\n")

# Check missing values in gee features dataset
missing_values = gee_features.isnull().sum()
print(missing_values)







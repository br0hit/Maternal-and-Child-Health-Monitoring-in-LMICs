import pandas as pd

# Load the gee_features dataset
gee_features = pd.read_csv('../data/gee_features.csv', nrows=10000)

# Load the sample submission file
sample_submission = pd.read_csv('../data/sample_submission.csv')

# Load the training labels
training_labels = pd.read_csv('../data/training_label.csv')

print(gee_features['DHSID'])
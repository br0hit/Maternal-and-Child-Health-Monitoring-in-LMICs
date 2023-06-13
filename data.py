import pandas as pd

# Load the gee_features dataset
gee_features = pd.read_csv('../data/gee_features.csv', nrows=100)
print('gee_features shape:', gee_features.shape)


# Load the sample submission file
sample_submission = pd.read_csv('../data/sample_submission.csv')

# Load the training labels
training_labels = pd.read_csv('../data/training_label.csv')

# Print the shape of each dataset to verify the number of records and features
print('sample_submission shape:', sample_submission.shape)
print('training_labels shape:', training_labels.shape)

print("Sample Submission Columns:")
print(sample_submission.columns)

print("\nTraining Labels Columns:")
print(training_labels.columns)

print("\nGee features Labels Columns:")
print(gee_features.columns)

# print("\nSample Submission Data Types:")
# print(sample_submission.dtypes)

print("\nTraining Labels Data Types:")
print(training_labels.dtypes)

print("\nGee features Data Types:")
print(gee_features.dtypes)

# print("Sample Submission Summary:")
# print(sample_submission.describe())

print("\nTraining Labels Summary:")
print(training_labels.describe())

print("\nGee Features Summary:")
print(gee_features.describe())

# print("Sample Submission Missing Values:")
# print(sample_submission.isnull().sum())

print("\nTraining Labels Missing Values:")
print(training_labels.isnull().sum())


print("\Gee features Missing Values:")
print(gee_features.isnull().sum())



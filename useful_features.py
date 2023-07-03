import pandas as pd

# Define the columns of interest from gee_features.csv
columns_of_interest = ['DHSID', 'DHSYEAR', 'DHSCLUST', 'LATNUM', 'LONGNUM','URBAN_RURA']

# Load the sample_submission.csv file
sample_submission = pd.read_csv('../data/sample_submission.csv')

# Extract the DHSID column
dhsids = sample_submission['DHSID']

# Read the selected columns directly from gee_features.csv
filtered_gee_features = pd.read_csv('../data/gee_features.csv', usecols=columns_of_interest)

# Filter based on the DHSIDs in sample_submission
filtered_gee_features = filtered_gee_features[filtered_gee_features['DHSID'].isin(dhsids)]

# Reorder the columns
filtered_gee_features = filtered_gee_features[['DHSID', 'DHSYEAR', 'DHSCLUST', 'LATNUM', 'LONGNUM','URBAN_RURA']]

# Save the filtered data to useful_features.csv
filtered_gee_features.to_csv('data/useful_features.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Load the training_label.csv file
data = pd.read_csv('../data/training_label.csv')

# Select the columns with missing values
columns_with_missing = ['Mean_BMI', 'Median_BMI']

# Create a new DataFrame with only the columns of interest
df = data[['DHSID', 'DHSYEAR', 'DHSCLUST', 'LATNUM', 'LONGNUM', 'Mean_BMI', 'Median_BMI', 'Unmet_Need_Rate', 'Under5_Mortality_Rate', 'Skilled_Birth_Attendant_Rate', 'Stunted_Rate', 'URBAN_RURA']].copy()

# Separate the rows with missing values from the rows with known values
known_data = df.dropna().reset_index(drop=True)  # Reset the index
unknown_data = df[df.isnull().any(axis=1)].reset_index(drop=True)  # Reset the index

# Prepare the features and target for the KNN imputation
X = known_data[['LATNUM', 'LONGNUM']]
y = known_data[columns_with_missing]

# Initialize the KNN imputer
imputer = KNNImputer(n_neighbors=3)  # You can adjust the number of neighbors (K) as per your needs

# Perform the KNN imputation on the unknown data
imputed_values = imputer.fit_transform(X, y)

# Create a DataFrame with the imputed values
imputed_data = pd.DataFrame(imputed_values, columns=columns_with_missing)

# Replace the missing values in the original DataFrame
unknown_data[columns_with_missing] = imputed_data

# Concatenate the known and imputed data
imputed_data = pd.concat([known_data, unknown_data], ignore_index=True)

# Save the imputed data to a new CSV file
imputed_data.to_csv('../data/imputed_data.csv', index=False)

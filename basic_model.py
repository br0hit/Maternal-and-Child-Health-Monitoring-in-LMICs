import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
    from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("../data/training_label.csv",nrows=340)

# Remove rows with NaN values
data = data.dropna()

# Encode the categorical 'DHSID' column
label_encoder = LabelEncoder()
data['DHSID'] = label_encoder.fit_transform(data['DHSID'])

x = data[['DHSID', 'DHSYEAR', 'DHSCLUST', 'LATNUM', 'LONGNUM']]
y = data[['Mean_BMI', 'Median_BMI']]  # Output features: Mean_BMI and Median_BMI

# print(x)
# print(y)

# Split the data into training and validation sets (80% for training, 20% for validation)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the models and their respective hyperparameter grids for tuning
models = {
    'Ridge Regression': (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
    'Lasso Regression': (Lasso(), {'alpha': [0.1, 1.0, 10.0]}),
    'SVR': (SVR(), {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}),
    'Gradient Boosting': (GradientBoostingRegressor(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.01]})
}

# Train and evaluate each model with hyperparameter tuning and cross-validation
for model_name, (model, param_grid) in models.items():
    # Perform grid search cross-validation
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best model and its predictions on the validation set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    
    # Evaluate the model
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Print the results
    print(f"Model: {model_name}")
    print("Best Parameters:", grid_search.best_params_)
    print("Mean squared error:", mse)
    print("R-squared:", r2)
    print()
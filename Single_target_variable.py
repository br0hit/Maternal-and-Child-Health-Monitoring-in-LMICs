import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("../data/training_label.csv", nrows=340)

# Remove rows with NaN values
data = data.dropna()

# Encode the categorical 'DHSID' column
label_encoder = LabelEncoder()
data['DHSID'] = label_encoder.fit_transform(data['DHSID'])

x = data[['DHSID', 'DHSYEAR', 'DHSCLUST', 'LATNUM', 'LONGNUM','Median_BMI']]
y = data['Mean_BMI']  # Selecting "Mean_BMI" as the target variable

# Split the data into training and validation sets (80% for training, 20% for validation)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the models and their parameter grids for hyperparameter tuning
models = [
    {
        'name': 'Ridge Regression',
        'model': Ridge(),
        'param_grid': {'alpha': [0.1, 1.0, 10.0]}
    },
    {
        'name': 'Lasso Regression',
        'model': Lasso(),
        'param_grid': {'alpha': [0.1, 1.0, 10.0]}
    },
    {
        'name': 'SVR',
        'model': SVR(),
        'param_grid': {'kernel': ['linear', 'rbf'], 'C': [1.0, 10.0]}
    },
    {
        'name': 'Gradient Boosting',
        'model': GradientBoostingRegressor(),
        'param_grid': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.5]}
    }
]

# Train and evaluate the models
for model_info in models:
    print(f"Model: {model_info['name']}")
    
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(model_info['model'], model_info['param_grid'], cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")
    
    # Make predictions on the validation set
    y_pred = best_model.predict(X_val)
    
    # Evaluate the model
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print("Mean squared error:", mse)
    print("R-squared:", r2)
    print()




######################

# C:\Users\Guestuser\anaconda3\lib\site-packages\sklearn\linear_model\_coordinate_descent.py:631: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.831e+01, tolerance: 3.538e-02
#   model = cd_fast.enet_coordinate_descent(
# Best Parameters: {'alpha': 1.0}
# Mean squared error: 2.0712452320365746
# R-squared: 0.05246429184944601

# Model: SVR
# Best Parameters: {'C': 1.0, 'kernel': 'rbf'}
# Mean squared error: 2.109881377026138
# R-squared: 0.0347893558077601

# Model: Gradient Boosting
# Best Parameters: {'learning_rate': 0.1, 'n_estimators': 100}
# Mean squared error: 1.849902871916451
# R-squared: 0.153722118154213

######################
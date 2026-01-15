import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml


# Load the House Prices dataset from OpenML as a pandas DataFrame
data = fetch_openml(name="house_prices", as_frame=True)

# Feature matrix (X): all input columns describing the houses
X = data.data

# Target vector (y): house prices; cast to float for numeric computation
y = data.target.astype(np.float32)


# Replace missing numeric values with the column median
# (simple, robust baseline imputation)
X = X.fillna(X.median(numeric_only=True))

# Convert categorical (string) columns into numeric one-hot encoded columns
# Example: Neighborhood = ['NAmes', 'CollgCr'] → Neighborhood_NAmes, Neighborhood_CollgCr
# drop_first=True avoids redundant columns and reduces multicollinearity
X = pd.get_dummies(X, drop_first=True)

# Split data into training and test sets
# Training set: used to fit the model
# Test set: used only to evaluate performance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,   # 20% held out for evaluation
    random_state=42   # fixed seed for reproducibility
)

# Random Forest regression model
# - n_estimators: number of trees
# - max_depth: limits tree complexity to reduce overfitting
# - min_samples_leaf: enforces minimum samples per leaf for smoother predictions
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1         # use all CPU cores
)

# Train the model on the training data
model.fit(X_train, y_train)

# Generate predictions for unseen test data
y_pred = model.predict(X_test)


# Mean Squared Error: average of squared prediction errors
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error: error in the same unit as the target (dollars)
rmse = np.sqrt(mse)

print(f"RMSE: {rmse:.4f}")

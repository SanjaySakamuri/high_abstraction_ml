import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml

data = fetch_openml(name="house_prices", as_frame=True)

X = data.data
y = data.target.astype(np.float32)
y = np.log1p(y)

X = X.fillna(X.median(numeric_only=True))
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

cv_scores = cross_val_score(
    model, X, y, cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

print("CV RMSE (log scale):", np.mean(np.sqrt(-cv_scores)))

model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)

y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

print("Test RMSE (dollar scale):", rmse)
print("OOB R^2:", model.oob_score_)

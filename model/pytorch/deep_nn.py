import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

X, y = fetch_california_housing(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state = 42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1) # Reshape for compatibility

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test =torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


## Scaling is done or else gradients will explode


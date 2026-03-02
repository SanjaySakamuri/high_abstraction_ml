import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# 1. Generate Dataset

X, y = make_classification(
    n_samples=2000, 
    n_features=20,
    n_classes=2,
    n_informative=15,
    random_state=42
)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0., random_state=42
)


# 3. Define Model

class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )

        def forward(self, x):
            return self.net(x)
        
model = MLP()


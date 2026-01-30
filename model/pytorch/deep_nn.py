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

class HousingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),  #Wide layer to capture complex patterns # 256 makes the results better. Maybe or maynot be a true value..
            nn.ReLU(), # ReLU activation for non-linearity

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
    
model = HousingNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr = 1e-3,
    weight_decay=1e-4 # L2 regularization
)

epochs = 500

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    loss.backward()
    optimizer.step()


    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    test_preds = model(X_test)

rmse = np.sqrt(mean_squared_error(y_test.numpy(), test_preds.numpy()))
print(f"Test RMSE: {rmse:.4f}")

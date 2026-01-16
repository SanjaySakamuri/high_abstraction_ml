import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 1. Reproducibility

torch.manual_seed(42)
np.random.seed(42)

# 2.Load & preprocess dataset

data = fetch_california_housing()
X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 3. Custom Dataset

class CaliforniaHousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = CaliforniaHousingDataset(X_train, y_train)
test_dataset = CaliforniaHousingDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)



# 4. Linear Regression Model

class LinearRegressionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)
    

model = LinearRegressionModel(n_features=X_train.shape[1])


# 5. Loss & Optimizer

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 6. Training Loop (Explicit)

epochs = 100

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0


    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)

    epoch_loss /= len(train_loader.dataset)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train MSE: {epoch_loss:.4f}")

# 7. Evaluation (No Grad)

model.eval()
test_loss = 0.0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        test_loss += loss.item() * X_batch.size(0)

test_loss /= len(test_loader.dataset)
print(f"\nTest MSE: {test_loss:.4f}")

# 8. Inspect Learned Weights

weights = model.linear.weight.data.numpy()
bias = model.linear.bias.data.item()

print("\nLearned weights:", weights)
print("Learned bias:", bias)


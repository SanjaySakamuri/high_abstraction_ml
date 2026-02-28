import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

torch.manual_seed(42)
np.random.seed(42)

# Load
X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_val   = torch.tensor(X_val, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.float32).view(-1,1)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

train_loader = DataLoader(TensorDataset(X_train,y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,y_val), batch_size=64)

class HousingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        return self.net(x)

model = HousingNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

best_val = float('inf')
patience = 20
counter = 0

epochs = 300

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for xb,yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds,yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb,yb in val_loader:
            preds = model(xb)
            loss = criterion(preds,yb)
            val_loss += loss.item() * xb.size(0)

    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

model.load_state_dict(torch.load("best_model.pt"))

model.eval()
with torch.no_grad():
    preds = model(X_test)

rmse = np.sqrt(mean_squared_error(y_test.numpy(), preds.numpy()))
print(f"Test RMSE: {rmse:.4f}")

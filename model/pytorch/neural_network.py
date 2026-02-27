import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

data = np.loadtxt("YearPredictionMSD.txt", delimiter=",")

y = data[:, 0]
X = data[:, 1:]

X_train, X_test = X[:463715], X[463715:]
y_train, y_test = y[:463715], y[463715:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class MSDDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_full_ds = MSDDataset(X_train, y_train)
test_ds = MSDDataset(X_test, y_test)

val_size = int(0.1 * len(train_full_ds))
train_size = len(train_full_ds) - val_size
train_ds, val_ds = random_split(train_full_ds, [train_size, val_size])  # FIX: added proper validation split

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)

class MSDNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MSDNet(input_dim=90)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # FIX: adaptive learning rate scheduler
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for Xb, yb in loader:
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(Xb)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for Xb, yb in loader:
            preds = model(Xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * len(Xb)
    return total_loss / len(loader.dataset)

for epoch in range(30):
    train_mse = train_epoch(model, train_loader)
    val_mse = eval_epoch(model, val_loader)
    scheduler.step(val_mse)  # FIX: scheduler now uses validation loss
    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | Train RMSE: {train_mse**0.5:.2f} | Val RMSE: {val_mse**0.5:.2f}")

test_mse = eval_epoch(model, test_loader)
print(f"Final Test RMSE: {test_mse**0.5:.2f}")

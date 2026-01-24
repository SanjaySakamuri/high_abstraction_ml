import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset. DataLoader
from sklearn.preprocessing import StandardScaler

data = np.loadtxt("YearPredictionMSD.txt", delimiter=",")

y = data[:,0]
X = data[:, 1:]

X_train, X_test = X[:463715], X[463715:]
y_train, y_test = y[:46715], X[463715:]

# Without scaling gradients explode !!!

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class MSDDataset(Dataset):
  def __init__(self, X,y):
    self.X = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]


train_ds = MSDDataset(X_train, y_train)
test_ds = MSDDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1024)

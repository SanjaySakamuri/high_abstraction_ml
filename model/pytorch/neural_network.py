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


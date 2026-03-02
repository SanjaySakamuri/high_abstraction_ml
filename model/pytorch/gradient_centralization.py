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

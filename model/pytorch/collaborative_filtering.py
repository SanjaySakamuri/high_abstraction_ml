import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

num_users = 1000
num_items = 500

np.random.seed(42)

data = {
    "user_id": np.random.randint(0, num_users, 10000),
    "item_id:": np.random.randint(0, num_items, 10000),
    "rating": np.random.randint(1, 6, 10000) # rating 1-5
}

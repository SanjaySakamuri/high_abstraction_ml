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

df = pd.DataFrame(data)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

class RecDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
    
train_dataset = RecDataset(train_df)
test_dataset = RecDataset(test_df)

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCF, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        x = torch.cat([user_emb, item_emb], dim=1)
        out = self.fc_layers(x)

        return out.squeeze()

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
    "item_id": np.random.randint(0, num_items, 10000),
    "rating": np.random.randint(1, 6, 10000)
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=1)
        out = self.fc_layers(x)
        return out.squeeze() * 4 + 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NCF(num_users, num_items).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader):
    model.train()
    total_loss = 0
    for user, item, rating in loader:
        user = user.to(device)
        item = item.to(device)
        rating = rating.to(device)
        optimizer.zero_grad()
        predictions = model(user, item)
        loss = criterion(predictions, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for user, item, rating in loader:
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)
            predictions = model(user, item)
            loss = criterion(predictions, rating)
            total_loss += loss.item()
    return total_loss / len(loader)

epochs = 10

for epoch in range(epochs):
    train_loss = train(model, train_loader)
    test_loss = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

def recommend(model, user_id, top_k=5):
    model.eval()
    user_tensor = torch.tensor([user_id] * num_items).to(device)
    item_tensor = torch.arange(num_items).to(device)
    with torch.no_grad():
        scores = model(user_tensor, item_tensor)
    top_items = torch.topk(scores, top_k).indices.cpu().numpy()
    return top_items

print(recommend(model, user_id=10))

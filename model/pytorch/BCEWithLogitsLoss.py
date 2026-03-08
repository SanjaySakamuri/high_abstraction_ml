import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.linear(x)
        return logits 
    
# Training Function

def train_epoch(model, dataloader, criterion, optimizer, device):
    
    model.train()
    total_loss = 0

    for X,y in dataloader:

        X = X.to(device)
        y = y.to(device).float().view(-1, 1)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(logits, y)
        
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in dataloader:

            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).long()

            correct += (preds.view(-1) == y).sum().item()
            total += y.size(0)

    return correct / total
# to do

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
        
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total


scaler = torch.cuda.amp.GradScaler()

def amp_train_step(model, batch, optimizer, criterion, device):
    model.train()
    x, y = batch
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        logits = model(x)
        loss = criterion(logits, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()
def custom_loss(logits, targets, alpha=0.01):
    ce = F.cross_entropy(logits, targets)
    reg = torch.mean(torch.norm(logits, dim=1))
    return ce + alpha * reg

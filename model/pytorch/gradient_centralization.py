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

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Define Model

class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )

        def forward(self, x):
            return self.net(x)
        
model = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. Gradient Centralization

def gradient_centralization(model):
    for param in model.parameters():
        if param.grad is not None and len(param.grad.shape) > 1:
            grad_mean = param.grad.mean(
                dim=tuple(range(1, param.grad.dim())),
                keepdim=True
            )
            param.grad -= grad_mean
# 4. Training Loop

epochs = 30
for epoch in range(epochs):
     model.train()

     optimizer.zero_grad()
     outputs = model(X_train)
     loss = criterion(outputs, y_train)
     loss.backward()

     # Apply Gradient Centralization
     gradient_centralization(model)

     optimizer.step()

     # Evaluation
     model.eval()
     with torch.no_grad():
          test_outputs = model(X_test)
          preds = torch.argmax(test_outputs, dim=1)
          accuracy = (preds == y_test).float().mean()
          print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}")

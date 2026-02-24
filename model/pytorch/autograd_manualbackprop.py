import torch
# Input and target

x = torch.tensor([2.0])
y_true = torch.tensor([8.0])

# parameters
w = torch.tensor([1.0], requires_grad=True) # requires_grad=True to compute gradients
b = torch.tensor([0.0], requires_grad=True)

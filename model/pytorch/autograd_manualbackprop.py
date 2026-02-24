import torch
# Input and target

x = torch.tensor([2.0])
y_true = torch.tensor([8.0])

# parameters
w = torch.tensor([1.0], requires_grad=True) # requires_grad=True to compute gradients
b = torch.tensor([0.0], requires_grad=True)

# forward pass: compute predicted y
y_pred = w * x + b

# Loss function
loss = (y_pred - y_true) ** 2

# backward pass: compute gradients
loss.backward()

print("Gradienty of w:", w.grad)  # Gradient of w
print("Gradienty of b:", b.grad)  # Gradient of b

learning_rate = 0.1

with torch.no_grad(): # Disable gradient tracking for the update step
    w -= learning_rate * w.grad # Update w
    b -= learning_rate * b.grad  # Update b

# Clear gradients for the next iteration
w.grad.zero_()
b.grad.zero_()

print(loss.grad_fn)  # Should be None, as loss is a scalar and does not have a gradient function

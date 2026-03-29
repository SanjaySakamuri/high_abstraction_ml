import numpy as np

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay

        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
      for i, (p, g) in enumerate(zip(self.params, grads)):
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
          
            # Adam update
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Decoupled weight decay
            p -= self.lr * self.wd * p

'''
decay = []
no_decay = []

for name, param in model.named_parameters():
    if "bias" in name or "LayerNorm.weight" in name:
        no_decay.append(param)
    else:
        decay.append(param)

optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 1e-2},
    {"params": no_decay, "weight_decay": 0.0}
], lr=1e-3)
'''

import numpy as np

class AdamW:
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        weight_decay_mask=None
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay

        self.wd_mask = weight_decay_mask if weight_decay_mask else [True] * len(params)

        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, lr=None, max_grad_norm=None):
        self.t += 1
        lr = lr if lr is not None else self.lr

        if max_grad_norm is not None:
            total_norm = np.sqrt(sum((g**2).sum() for g in grads))
            clip_coef = max_grad_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                grads = [g * clip_coef for g in grads]

        for i, (p, g) in enumerate(zip(self.params, grads)):

            if g.ndim > 1:
                g = g - g.mean(axis=tuple(range(1, g.ndim)), keepdims=True)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            if self.wd_mask[i]:
                p -= lr * self.wd * p

            p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self):
        return {
            "m": self.m,
            "v": self.v,
            "t": self.t
        }

    def load_state_dict(self, state):
        self.m = state["m"]
        self.v = state["v"]
        self.t = state["t"]

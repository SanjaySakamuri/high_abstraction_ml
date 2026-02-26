import torch
from torch.optim.optimizer import Optimizer

class Adam(Optimizer):
    """
    Implements Adam optimization algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to denominator for numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps <= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1 parameter")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2 parameter")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                step = state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group["eps"])

                step_size = group["lr"] / bias_correction1

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

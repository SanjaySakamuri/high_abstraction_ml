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

import numpy as np

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.nn import functional as F


class Dice(nn.Module):
    """
    Sørensen-Dice Coefficient as a loss function. Sudre C. et al.
    "`Generalised Dice overlap as a deep learning loss function for highly
    unbalanced segmentations <https://arxiv.org/abs/1707.03237>`_".

    Predictions are passed through a softmax function to obtain probabilities.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):
        """
        Forward method

        Note
        ----
            class 255 is deleted out of calculation.
        """
        num_classes = prediction.shape[1]
        ndims = target.ndimension()

        target = torch.where(target == 255, 0, target)

        prediction = F.softmax(prediction, dim=1)

        target = target.long()
        target = torch.eye(num_classes, device=target.device)[target.squeeze(1)]
        target = target.permute(0, -1, *tuple(range(1, ndims - 1))).float()
        target = target.to(prediction.device).type(prediction.type())

        dims = (0,) + tuple(range(2, ndims))
        intersection = torch.sum(prediction * target, dims)
        cardinality = torch.sum(prediction + target, dims)
        dice_coeff = (2. * intersection / (cardinality + self.eps)).mean()
        return 1 - dice_coeff


class Lion(Optimizer):
    """
    Implements Lion algorithm. X Chen, C Liang et.al "Symbolic discovery of optimization algorithms"
    https://arxiv.org/abs/2302.06675
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """
        Initialize the hyperparameters.

        Parameters
        ----------
          params: iterable
              iterable of parameters to optimize or dicts defining parameter groups
          lr: float
              learning rate (default: 1e-4)
          betas: Tuple[float, float]
              coefficients used for computing running averages of gradient and its square
              default: (0.9, 0.99)
          weight_decay: float
              weight decay coefficient (default: 0)
        """
        if lr <= 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Parameters
        ----------
          closure: callable
              A closure that reevaluates the model and returns the loss.

        Returns
        -------
            the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:

                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def mIoU(preds, mask, eps=1e-10, num_classes=150, multiclass='micro'):
    """
    Calculates mean intersection over union.
    Does not take into account background class and classes that are not in masks

    Parameters
    ----------

    preds: torch.tensor
        predictions tensor of shape (B, C, H, W) where C - n_classes
    mask: torch.tensor
        masks tensor of shape (B, H, W)
    eps: float
        a number that is added so as not to divide by zero
    n_classes: int
        number of classes

    Returns
    -------
    float
        mean mIoU
    """
    assert multiclass in ['micro', 'macro']

    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)

    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    preds = preds.to(torch.device('cuda:0'))
    mask = mask.to(torch.device('cuda:0'))

    preds = torch.where(preds == 255, 0, preds)
    mask = torch.where(mask == 255, 0, mask)

    batch_size = preds.size(0)

    with torch.no_grad():
        preds = F.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)

        if multiclass == 'micro':
            preds = preds.contiguous().view(-1)
            mask = mask.contiguous().view(-1)

            dimension = 0
        else:
            preds = preds.contiguous().view(batch_size, -1) # B, H*W
            mask = mask.contiguous().view(batch_size, -1) # B, H*W

            dimension = 1

        one_hot_preds = F.one_hot(preds, num_classes=num_classes)
        one_hot_mask = F.one_hot(mask, num_classes=num_classes)

        presence_of_class = (one_hot_preds.any(dim=dimension) + one_hot_mask.any(dim=dimension)) > 0

        if multiclass == 'micro':
            presence_of_class[0] = False # doesn't take into account background
        else:
            presence_of_class[:, 0] = False

        intersect = torch.logical_and(one_hot_preds, one_hot_mask).sum(dim=dimension)
        union = torch.logical_or(one_hot_preds, one_hot_mask).sum(dim=dimension)

        iou = torch.where(presence_of_class, ((intersect + eps)/(union + eps)), np.nan)
        classes_miou = torch.nanmean(iou, dim=dimension)
        return torch.mean(classes_miou).detach().cpu().numpy()

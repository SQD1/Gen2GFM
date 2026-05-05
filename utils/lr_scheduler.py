from torch.optim.lr_scheduler import LRScheduler
import math
import torch
from torch.optim.optimizer import Optimizer

class WarmupCosineDecayScheduler(LRScheduler):
    """
    Scheduler that first linearly warms up from 0 to max_lr (over warmup_epochs),
    then decays from max_lr to min_lr following a cosine curve from epoch warmup_epochs
    to total_epochs.

    By default, you call .step() once per epoch. The 'epoch' is tracked by self.last_epoch.
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_lr,
        min_lr,
        total_epochs,
        last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.init_lr = 2e-5
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        This function returns a list of learning rates for each param_group in the optimizer.
        It uses self.last_epoch (0-based) to decide if we are in warmup or cosine decay phase.
        """
        epoch = self.last_epoch  # 0-based epoch index

        if epoch >= self.total_epochs:
            epoch = self.total_epochs

        if epoch < self.warmup_epochs:
            lr = self.init_lr + (self.max_lr - self.init_lr) * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / float(self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine

        return [lr for _ in self.optimizer.param_groups]


def MultiStepLR(
    optimizer: Optimizer, total_iters: int, lr_milestones: list[float]
) -> LRScheduler:
    
    """
    This module provides a utility function for creating a multi-step learning rate scheduler.

    Functions:
        MultiStepLR(optimizer: Optimizer, total_iters: int, lr_milestones: list[float]) -> LRScheduler
            Creates a MultiStepLR scheduler with specified milestones and decay factor.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        total_iters (int): The total number of iterations for training.
        lr_milestones (list[float]): A list of fractions representing the milestones at which the learning rate will be decayed.

    Returns:
        LRScheduler: A PyTorch learning rate scheduler that decays the learning rate at specified milestones.
    """

    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [int(total_iters * r) for r in lr_milestones], gamma=0.1
    )
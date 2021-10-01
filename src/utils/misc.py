import time

import gin
import numpy as np
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_triplet(high, size):
    triplets = torch.randint(low=0, high=high, size=(int(size * 1.2), 3))
    local_dup_check = (triplets - triplets.roll(1, 1) != 0).all(dim=1)
    triplets = triplets[local_dup_check]
    return triplets


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0
        self.max = 0
        self.min = np.inf

    def update(self, val, n=1):
        if isinstance(val, np.ndarray):
            n = val.size
            val = val.mean()
        elif isinstance(val, torch.Tensor):
            n = val.nelement()
            val = val.mean().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


class Timer(AverageMeter):
    """A simple timer."""

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.update(self.diff)
        if average:
            return self.avg
        else:
            return self.diff


@gin.configurable()
def logged_hparams(keys):
    C = dict()
    for k in keys:
        C[k] = gin.query_parameter(f"%{k}")
    return C

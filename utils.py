import random
from typing import Any, List, Optional, Callable

import numpy as np
import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def args_type(default: Any) -> Any:
    def parse_string(x: str):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def attrdict_factory(
    metrics: Optional[List[str]] = None, template: Callable = AverageMeter
) -> AttrDict:
    """Convenience function for generating an AttrDict object with arbitrary keys initialized using a factory
    function, essentially an AttrDict extension to collections.defaultdict. 'metrics' is a list of keys (usually
    strings) and the initial value of these keys can optionally be set by passing an object factory to the parameter
    'template', which defaults to naslib.utils.utils.AverageMeter."""

    metrics = AttrDict({m: template() for m in metrics} if metrics else {})
    return metrics


def seed_all(seed: int):
    """Set the seed for torch, np, python random
    Args:
        seed (int): seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

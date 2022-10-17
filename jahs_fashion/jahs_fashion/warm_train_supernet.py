import random
import time

import numpy as np
import torch

from model.dynamic_model import DynaJAHSBench201Network
from train import train_subnet
import utils

# Script to warm start the supernet by training it for 10 epochs
if __name__ == "__main__":
    config_1x1 = {
        "Optimizer": "SGD",
        "LearningRate": 0.1,
        "WeightDecay": 0.001,
        "Activation": "ReLU",
        "TrivialAugment": True,
        "Op1": 3,
        "Op2": 3,
        "Op3": 3,
        "Op4": 3,
        "Op5": 3,
        "Op6": 3,
        "N": 5,
        "W": 16,
        "Resolution": 1.0,
    }
    config_3x3 = {
        "Optimizer": "SGD",
        "LearningRate": 0.1,
        "WeightDecay": 0.001,
        "Activation": "ReLU",
        "TrivialAugment": True,
        "Op1": 2,
        "Op2": 2,
        "Op3": 2,
        "Op4": 2,
        "Op5": 2,
        "Op6": 2,
        "N": 5,
        "W": 16,
        "Resolution": 1.0,
    }

    device = torch.device("cuda")
    seed = 101
    utils.seed_all(seed)

    supernet = DynaJAHSBench201Network()
    start_time = time.time()

    diverged, metrics, subnet, endtime = train_subnet(
        supernet, config_1x1, seed=101, device=device, budget=5, validate=True, test=True
    )
    runtime = endtime - start_time
    print(subnet is supernet)
    print(metrics)
    print(runtime)

    start_time2 = time.time()
    diverged, metrics, subnet, endtime2 = train_subnet(
        subnet, config_3x3, seed=101, device=device, budget=5, validate=True, test=True
    )
    runtime2 = endtime2 - start_time2
    print(subnet is supernet)
    print(metrics)
    print(runtime2 + runtime)
    torch.save(subnet.state_dict(), "warmSupernet2.state")

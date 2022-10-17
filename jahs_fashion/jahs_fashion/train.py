import copy
import logging
import time
from typing import Any, Callable, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from ConfigSpace import Configuration
from scipy.io.idl import AttrDict
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from jahs_bench.tabular.lib.core import datasets as dataset_lib
from jahs_bench.tabular.lib.core.constants import Datasets
from torch.utils.data import DataLoader

from utils import  accuracy, attrdict_factory


datadir = Path(__file__).parent.parent / "datasets"

def construct_model_optimizer(model: nn.Module, cfg: Configuration) -> Tuple[Optimizer, _LRScheduler, nn.CrossEntropyLoss]:
    """Construct the optimizer, scheduler and loss function
     for the given neural net
    Args:
        model (nn.Module): the neural net
        cfg (Configuration): the configuration for optimizer

    Returns:
        Tuple[Optimizer, CosineAnnealingLR, nn.CrossEntropyLoss]: the constructed objects
    """
    optimizer = SGD(
        model.parameters(),
        momentum=0.9,
        lr=cfg["LearningRate"],
        weight_decay=cfg["WeightDecay"],
        nesterov=True,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0.0)

    loss_fn = nn.CrossEntropyLoss()

    return optimizer, scheduler, loss_fn


def _main_proc(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    mode: str,
    device: torch.device,
    epoch_metrics: Dict[str, Any],
    use_grad_clipping: bool = True,
) -> Tuple[bool, nn.Module]:
    """The main training process

    Args:
        model (nn.Module): The neural net
        dataloader (DataLoader): data loader
        loss_fn (Callable): Loss function
        optimizer (Optimizer): Optimizer
        scheduler (_LRScheduler): Learning rate scheduler
        mode (str): train/eval
        device (torch.device): device
        epoch_metrics (Dict[str, Any]): the metrics dict for each epoch
        use_grad_clipping (bool, optional): Defaults to True.

    Raises:
        ValueError: For wrong mode

    Returns:
        Tuple[bool, nn.Module]: Bool indicating if training diverged and the Trained model
    """
    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()
    else:
        raise ValueError(f"Unrecognized mode '{mode}'.")

    train_model = mode == "train"

    dataset_metrics = ["loss", "acc"]
    metrics = attrdict_factory(metrics=dataset_metrics)
    ## Iterate over mini-batches
    diverged = False
    nsteps = len(dataloader)
    for step, (inputs, labels) in enumerate(dataloader):
        metric_weight = inputs.size(0)
        start_time = time.time()

        if train_model:
            optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)

        ## Forward Pass
        logits = model(inputs)
        loss = loss_fn(logits, labels)

        ## Backward Pass
        if train_model:
            loss.backward()
            if use_grad_clipping:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=2.0, norm_type=2
                )
            optimizer.step()

        metrics.loss.update(loss.detach().cpu().data.item(), metric_weight)
        acc = accuracy(logits.detach().cpu(), labels.detach().cpu(), topk=(1,))[0]
        metrics.acc.update(acc.data.item(), metric_weight)
        if torch.isnan(loss):
            diverged = True

        if diverged:
            return 0, True, None
    for key, value in metrics.items():
        epoch_metrics[key].append(value.avg)
    return False, model


def train_subnet(model: nn.Module, config: Configuration, seed: int, device: int, budget: int, validate: bool=True, test: bool=True, train: bool=True) -> Tuple[bool, Dict, nn.Module, int]:
    """Samples a model from supernet and fits the given data on it.
    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    Args:
        model (nn.Module): The supernet
        config (Configuration): Configuration of subnet and hyperparameters
        seed (int): seed
        device (int): torch device
        budget (int): number of epochs
        validate (bool, optional): Run trained model on validation set. Defaults to True.
        test (bool, optional): Run trained model on the test set. Defaults to True.
        train (bool, optional): Whether the model needs to be trained. Used to Evaluate. Defaults to True.

    Returns:
        Tuple[bool, Dict, nn.Module, int]: Boolean indicating if training diverged, metrics, trained model, endtime
    """
    end_time = time.time()
    lr = config["LearningRate"]
    weight_decay = config["WeightDecay"]
    resolution = config["Resolution"]
    trivial_augment = config["TrivialAugment"]
    optimizer = config["Optimizer"]

    batch_size = 256

    # Device configuration
    torch.manual_seed(seed)
    num_epochs = int(np.ceil(budget))

    model.set_sample_cfg(config)
    optimizer, scheduler, loss_fn = construct_model_optimizer(model, config)
    loss_fn = loss_fn.to(device)
    model.to(device)

    data_loaders, min_shape = dataset_lib.get_dataloaders(
        dataset=Datasets.fashion_mnist,
        batch_size=256,
        cutout=0,
        split=validate,
        resolution=resolution,
        trivial_augment=trivial_augment,
        datadir=datadir,
    )

    train_queue = data_loaders["train"]
    test_queue = data_loaders["test"]
    if validate:
        valid_queue = data_loaders["valid"]

    dataset_metrics = ["loss", "acc"]
    model_metrics = AttrDict(
        {
            "train": attrdict_factory(metrics=dataset_metrics, template=list),
            "valid": attrdict_factory(metrics=dataset_metrics, template=list),
            "test": attrdict_factory(metrics=dataset_metrics, template=list),
        }
    )

    train_size = valid_size = test_size = 0

    diverged = False

    if train:
        for e in tqdm(range(1, budget + 1)):
            ## Handle training set
            dataloader = train_queue
            epoch_metrics = model_metrics.train
            diverged, subnet = _main_proc(
                model=model,
                dataloader=dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                mode="train",
                device=device,
                epoch_metrics=epoch_metrics,
            )
            scheduler.step()
            if diverged:
                break

        ## Handle validation set, if needed
    if validate:
        val_time = time.time()
        dataloader = valid_queue
        epoch_metrics = model_metrics.valid
        with torch.no_grad():
            _, _ = _main_proc(
                model=model,
                dataloader=dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                mode="eval",
                device=device,
                epoch_metrics=epoch_metrics,
            )
        end_time = time.time()
        print(end_time - val_time)

    ## Handle test set, if needed
    if test:
        test_time = time.time()
        dataloader = test_queue
        epoch_metrics = model_metrics.test
        with torch.no_grad():
            _, _ = _main_proc(
                model=model,
                dataloader=dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                mode="eval",
                device=device,
                epoch_metrics=epoch_metrics,
            )
        print(time.time() - test_time)

        ## Checkpointing
        # Add a one-time offset to the runtime in case an old checkpoint was loaded

    if diverged:
        return True, None, model, end_time

    return False, model_metrics, model, end_time

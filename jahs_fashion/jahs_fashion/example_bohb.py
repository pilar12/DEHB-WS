"""
===========================
Optimization using BOHB
===========================
"""
import argparse
from functools import partial
import json
import logging

import numpy as np
import torch

import ConfigSpace as CS
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)
from sklearn.model_selection import StratifiedKFold

from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms

from cnn import torchModel


def get_optimizer_and_crit(cfg):
    if cfg["optimizer"] == "AdamW":
        model_optimizer = torch.optim.AdamW
    else:
        model_optimizer = torch.optim.Adam

    if cfg["train_criterion"] == "mse":
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss
    return model_optimizer, train_criterion


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
def cnn_from_cfg(cfg: Configuration, seed: int, instance: str, budget: float):
    """
    Creates an instance of the torch_model and fits the given data on it.
    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    :param cfg: Configuration (basically a dictionary)
        configuration chosen by smac
    :param seed: int or RandomState
        used to initialize the rf's random generator
    :param instance: str
        used to represent the instance to use (just a placeholder for this example)
    :param budget: float
        used to set max iterations for the MLP
    Returns
    -------
    val_accuracy cross validation accuracy
    """
    lr = cfg["learning_rate_init"] if cfg["learning_rate_init"] else 0.001
    batch_size = cfg["batch_size"] if cfg["batch_size"] else 200
    data_dir = cfg["data_dir"] if cfg["data_dir"] else "FashionMNIST"
    device = cfg["device"] if cfg["device"] else "cpu"

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    img_width = 28
    img_height = 28
    input_shape = (1, img_width, img_height)

    pre_processing = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_val = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=pre_processing
    )

    # returns the cross validation accuracy
    cv = StratifiedKFold(
        n_splits=3, random_state=42, shuffle=True
    )  # to make CV splits consistent
    num_epochs = int(np.ceil(budget))
    score = []

    # for train_idx, valid_idx in cv.split(data, data.targets):
    for train_idx, valid_idx in cv.split(train_val, train_val.targets):
        train_data = Subset(train_val, train_idx)
        val_data = Subset(train_val, valid_idx)
        train_loader = DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
        model = torchModel(
            cfg, input_shape=input_shape, num_classes=len(train_val.classes)
        ).to(model_device)

        summary(model, input_shape, device=device)

        model_optimizer, train_criterion = get_optimizer_and_crit(cfg)
        optimizer = model_optimizer(model.parameters(), lr=lr)
        train_criterion = train_criterion().to(device)

        for epoch in range(num_epochs):
            logging.info("#" * 50)
            logging.info("Epoch [{}/{}]".format(epoch + 1, num_epochs))
            train_score, train_loss = model.train_fn(
                optimizer, train_criterion, train_loader, model_device
            )
            logging.info("Train accuracy %f", train_score)

        val_score = model.eval_fn(val_loader, device)
        score.append(val_score)

    val_acc = 1 - np.mean(score)  # because minimize
    return val_acc


if __name__ == "__main__":
    """
    This is just an example of how to implement BOHB as an optimizer!
    Here we do not consider any of the forbidden clauses.
    """
    parser = argparse.ArgumentParser(description="JAHS")
    parser.add_argument("--data_dir", type=str, default="./FashionMNIST")
    parser.add_argument(
        "--working_dir",
        default="./tmp",
        type=str,
        help="directory where intermediate results are stored",
    )
    parser.add_argument(
        "--runtime",
        default=21600,
        type=int,
        help="Running time allocated to run the algorithm",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="maximal number of epochs to train the network",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to run the models"
    )

    args = parser.parse_args()

    logger = logging.getLogger("MLP-example")
    logging.basicConfig(level=logging.INFO)

    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()

    # We can add multiple hyperparameters at once:
    n_conf_layer = UniformIntegerHyperparameter("n_conv_layers", 1, 3, default_value=3)
    n_conf_layer_0 = UniformIntegerHyperparameter(
        "n_channels_conv_0", 512, 2048, default_value=512
    )
    n_conf_layer_1 = UniformIntegerHyperparameter(
        "n_channels_conv_1", 512, 2048, default_value=512
    )
    n_conf_layer_2 = UniformIntegerHyperparameter(
        "n_channels_conv_2", 512, 2048, default_value=512
    )

    learning_rate_init = UniformFloatHyperparameter(
        "learning_rate_init",
        0.00001,
        1.0,
        default_value=2.244958736283895e-05,
        log=True,
    )
    cs.add_hyperparameters(
        [
            n_conf_layer,
            n_conf_layer_0,
            n_conf_layer_1,
            n_conf_layer_2,
            learning_rate_init,
        ]
    )

    # Add conditions to restrict the hyperparameter space
    use_conf_layer_2 = CS.conditions.InCondition(n_conf_layer_2, n_conf_layer, [3])
    use_conf_layer_1 = CS.conditions.InCondition(n_conf_layer_1, n_conf_layer, [2, 3])
    # Add  multiple conditions on hyperparameters at once:
    cs.add_conditions([use_conf_layer_2, use_conf_layer_1])

    data_dir = args.data_dir
    runtime = args.runtime
    device = args.device
    max_epochs = args.max_epochs
    working_dir = args.working_dir

    cs.add_hyperparameters(
        [
            Constant("device", device),
            Constant("data_dir", data_dir),
        ]
    )
    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "wallclock-limit": runtime,  # max duration to run the optimization (in seconds)
            "cs": cs,  # configuration space
            "output-dir": working_dir,  # working directory where intermediate results are stored
            "deterministic": "True",
            # "limit_resources": True,  # Uses pynisher to limit memory and runtime
            # Then you should handle runtime and memory yourself in the TA
            # If you train the model on a CUDA machine, then you need to disable this option
            # "memory_limit": 8192,  # adapt this to reasonable value for your hardware
        }
    )

    # max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_epochs = 50
    # intensifier parameters (Budget parameters for BOHB)
    intensifier_kwargs = {"initial_budget": 5, "max_budget": max_epochs, "eta": 3}
    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=cnn_from_cfg,
        intensifier_kwargs=intensifier_kwargs,
        # all arguments related to intensifier can be passed like this
        initial_design_kwargs={
            "n_configs_x_params": 1,  # how many initial configs to sample per parameter
            "max_config_fracs": 0.2,
        },
    )
    # def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(),
    #                                       instance='1', budget=max_iters, seed=0)[1]
    # Start optimization
    try:  # try finally used to catch any interupt
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(
        config=incumbent, instance="1", budget=max_epochs, seed=0
    )[1]
    print("Optimized Value: %.4f" % inc_value)

    # srore your optimal configuration to disk
    opt_config = incumbent.get_dictionary()
    with open("opt_cfg.json", "w") as f:
        json.dump(opt_config, f)

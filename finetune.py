import os

import torch
from ConfigSpace import Configuration
from model.dynamic_model import DynaJAHSBench201Network
from train import train_subnet

import glob
import json


def finetune(config: Configuration, model_path: str, **kwargs):
    """Inheriting weights from the supernet with the incumbent configuration and finetuning and evaluation on the test
    set
        Args:
            config (Configuration): Architecture+Hparam config
            model_path (str): Supernet model path
            kwargs (Dict): dict with seed and device
        Returns:
            (dict) Model performance metrics
    """
    config = dict(config)
    config["N"] = 5
    config["W"] = 16
    config["Resolution"] = 1.0
    device = torch.device(kwargs["device"])
    seed = kwargs["seed"]
    model = DynaJAHSBench201Network()
    model.load_state_dict(torch.load(model_path))
    model.set_sample_cfg(config)
    finetune_epochs = 15
    diverged, metrics, model, _ = train_subnet(model, config, seed, device, finetune_epochs, validate=False)
    if not diverged:
        torch.save(model.state_dict(), os.path.dirname(model_path) + "/finetune" + str(finetune_epochs) + ".state")
        return metrics
    else:
        return 0


def inherit_test(config: Configuration, model_path: str, **kwargs):
    """Inheriting weights from the supernet with the incumbent configuration  and evaluation on the test set

    Args:
        config (Configuration): Architecture+Hparam config
        model_path (str): Supernet model path
        kwargs (Dict): dict with seed and device
    Returns:
        (dict) Model performance metrics
    """
    config = dict(config)
    config["N"] = 5
    config["W"] = 16
    config["Resolution"] = 1.0
    device = torch.device(kwargs["device"])
    seed = kwargs["seed"]
    model = DynaJAHSBench201Network()
    model.load_state_dict(torch.load(model_path))
    model.set_sample_cfg(config)
    diverged, metrics, _, _ = train_subnet(model, config, seed, device, 10, validate=False, train=False)
    if not diverged:
        torch.save(model.state_dict(), os.path.dirname(model_path) + "/inherit.state")
        return metrics
    else:
        return 0


# Script to compute performance of incumbent configurations
if __name__ == "__main__":
    for i in glob.glob("dehbws_results/*/")[1:]:
        print("*************************************")
        incumbent = glob.glob(i + "incumbent*")[0]
        with open(incumbent, "r+") as fp:
            incumbent = json.load(fp)["config"]
        model_path = glob.glob(i + "super*")[0]
        print(incumbent)
        print(model_path)
        seed = int(os.path.basename(os.path.dirname(model_path)))
        print("inherit performance")
        print(inherit_test(incumbent, model_path, seed=seed, device="cuda"))
        print("finetune performance")
        print(finetune(incumbent, model_path, seed=seed, device="cuda"))
        print("******************************************")

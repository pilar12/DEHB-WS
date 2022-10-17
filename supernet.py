import copy
import glob
import os
import shutil
from pathlib import Path
from typing import List

import torch
from ConfigSpace import Configuration

import utils
from model.dynamic_model import DynaJAHSBench201Network

modeldir = Path(__file__).parent / "tmp"
Supernet = None


def create_supernet(seed: int, warm: bool = False):
    """Creates the global supernet

    Args:
        seed (int): random seed
        warm (bool, optional): If warm start is true, we will load the warm started supernet
        on the dataset for 10 epochs. Defaults to False.
    """
    utils.seed_all(seed)
    global Supernet
    try:
        os.mkdir(modeldir)
    except:
        shutil.rmtree(modeldir)
        os.mkdir(modeldir)
    Supernet = DynaJAHSBench201Network()
    if warm:
        Supernet.load_state_dict(torch.load("model/warmSupernet.state"))
        print("Using warm supernet")


def get_path(config):
    """Set default config parameters and return the filepath of the model
    for a given config."""
    config = dict(config)
    config["N"] = 5
    config["W"] = 16
    config["Resolution"] = 1.0
    x = list(config.keys())
    x.sort()
    file_path = "tmp/"
    for i in x:
        file_path += str(config[i])
    return file_path


def update_func(configs: List[Configuration]):
    """Update the supernet for the given subnet configurations

    Args:
        configs (List[Configuration]): The subnet configurations
    """
    global Supernet
    m_update = []  # models for which to update the supernet
    m_file = glob.glob("tmp/*")
    for config in configs:
        path = get_path(config)
        if path in m_file:
            m_update.append(path)
    dict_params = None

    for i in m_update:
        tmp_state_dict = torch.load(i)
        if dict_params:
            for key in tmp_state_dict:
                dict_params[key] += tmp_state_dict[key] / float(len(m_update))
        else:
            dict_params = copy.deepcopy(tmp_state_dict)
            for key in tmp_state_dict:
                dict_params[key] = tmp_state_dict[key] / float(len(m_update))
    if dict_params:
        Supernet.load_state_dict(dict_params)
        print("Supernet Updated")
    shutil.rmtree(modeldir)
    os.mkdir(modeldir)

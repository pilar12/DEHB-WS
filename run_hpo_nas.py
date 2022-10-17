import argparse
import copy
import functools
import json
import logging
import os
import pathlib
import pickle
import sys
import time
from typing import Any, Dict

import jahs_bench
import numpy as np
import ruamel.yaml as yaml
import torch
from ConfigSpace import Configuration
from dehb import DEHB
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

import supernet
import utils
from configspace import get_configspace
from dehbws import DEHBWS
from model.dynamic_model import DynaJAHSBench201Network
from supernet import create_supernet, update_func, get_path
from train import train_subnet

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def train_and_evaluate(config: Configuration, budget: float, **kwargs) -> float:
    """Final training and evaluation on the test set

    Args:
        config (Configuration): Architecture+Hparam config
        budget (float): Number of epochs
        kwargs (Dict): dict with seed and device
    Returns:
        float: test accuracy
    """
    config = dict(config)
    config["N"] = 5
    config["W"] = 16
    config["Resolution"] = 1.0
    if kwargs["use_benchmark"]:
        global _BENCHMARK
        if _BENCHMARK is None:
            _BENCHMARK = jahs_bench.Benchmark(
                task="fashion_mnist", download=True, save_dir="./data/jahs"
            )
        num_epochs = int(np.ceil(budget))
        results = _BENCHMARK(config, nepochs=num_epochs)
        test_acc = results[num_epochs]["test-acc"] / 100.0
        cost = results[num_epochs]["runtime"]
        return test_acc

    device = torch.device(kwargs["device"])
    seed = kwargs["seed"]
    subnet = DynaJAHSBench201Network()
    budget = int(np.ceil(budget))
    diverged, metrics, subnet, _ = train_subnet(
        subnet, config, seed, device, budget, validate=False
    )
    if not diverged:
        return metrics.test.acc[-1]
    else:
        return 0


def dehb_objective_fn(config: Configuration, budget: float, **kwargs) -> Dict[str, Any]:
    """Objective function for dehb to optimize over
    Args:
        config (Configuration): Architecture+Hparam config
        budget (float): Number of epochs
        kwargs (Dict): dict with seed and device
    Returns:
        Dict[str, Any]: The cost in terms of valiation regret and the runtime
    formatted for dehb
    """
    results = objective_fn(config, budget, **kwargs)
    res = {
        "fitness": results["val_regret"],
        "cost": results["runtime"],
        "info": {
            "train_loss": results["train_loss"],
            "valid_loss": results["val_loss"],
            "test_loss": results["test_loss"],
            "train_regret": results["train_regret"],
            "test_regret": results["test_regret"],
        },
    }
    return res


def smac_objective_fn(cfg: Configuration, budget: float, **kwargs) -> float:
    """Objective function for smac to optimize over.
    We log the metadata for analysis as smac doesn't do that automatically.

    Args:
        cfg (Configuration): Architecture+Hparam config
        budget (float): Number of epochs
        kwargs (Dict): dict with seed and device
    Returns:
        float: validation regret
    """
    results = objective_fn(cfg, budget, **kwargs)
    dictionary = {
        "train_loss": results["train_loss"],
        "val_loss": results["val_loss"],
        "test_loss": results["test_loss"],
        "train_regret": results["train_regret"],
        "val_regret": results["val_regret"],
        "test_regret": results["test_regret"],
        "train_time": results["runtime"],
        "budget": int(np.ceil(budget)),
    }
    # Read JSON file
    history_path = pathlib.Path(kwargs["output_path"]) / "run_history.json"
    if history_path.exists():
        with history_path.open() as fp:
            list_obj = json.load(fp)
    else:
        list_obj = []
    list_obj.append(dictionary)

    with history_path.open("w") as f:
        json.dump(list_obj, f, indent=4, separators=(",", ": "))

    return results["val_regret"]


def objective_fn(config: Configuration, budget: float, **kwargs) -> Dict[str, float]:
    """The actual objective function for optimization
     where we either train the subnet model built using
     the config for "budget" number of epochs Or use the benchmark
     to obtain the cost (validation regret), runtime for training etc.

    Args:
        config (Configuration): Architecture+Hparam config
        budget (float): Number of epochs
        kwargs (Dict): dict with seed and device

    Returns:
        _type_: Dict with validation regret, test regret, runtime
     and other metrics for analysis.
    """
    config = dict(config)
    # Fixing constant config values here as there's a bug in DEHB
    config["N"] = 5
    config["W"] = 16
    config["Resolution"] = 1.0
    if kwargs["use_benchmark"]:
        return evaluate_config_from_benchmark(config, budget)
    if "effective_budget" in kwargs.keys() and kwargs["effective_budget"]:
        budget = kwargs["effective_budget"]
    device = torch.device(kwargs["device"])
    seed = kwargs["seed"]
    subnet = copy.deepcopy(supernet.Supernet)
    start_time = time.time()
    budget = int(np.ceil(budget))
    diverged, metrics, subnet, end_time = train_subnet(
        subnet, config, seed, device, budget
    )
    runtime = end_time - start_time
    train_acc = metrics.train.acc[-1]
    val_acc = metrics.valid.acc[-1]
    test_acc = metrics.test.acc[-1]

    if diverged:
        return {
            "runtime": runtime,
            "train_regret": 1,
            "train_loss": np.inf,
            "val_regret": 1,
            "val_loss": np.inf,
            "test_regret": 1,
            "test_loss": np.inf,
        }

    else:
        path = get_path(config)
        torch.save(subnet.state_dict(), path)
        logger.info("Saving temp model ...")
        return {
            "runtime": runtime,
            "train_regret": 1 - train_acc,
            "train_loss": metrics.train.loss[-1],
            "val_regret": 1 - val_acc,
            "val_loss": metrics.valid.loss[-1],
            "test_regret": 1 - test_acc,
            "test_loss": metrics.test.loss[-1],
        }


_BENCHMARK = None


def evaluate_config_from_benchmark(
    config: Configuration, budget: float
) -> Dict[str, float]:
    """Use JAHSbench to evaluate the config at a given budget
    Args:
        config (Configuration): Architecture+Hparam config
        budget (float): Number of epochs

    Returns:
        Dict[str, float]: dict with validation regret, test regret, runtime
     and other metrics for analysis.
    """
    global _BENCHMARK
    if _BENCHMARK is None:
        _BENCHMARK = jahs_bench.Benchmark(
            task="fashion_mnist", download=True, save_dir="./data/jahs"
        )
    num_epochs = int(np.ceil(budget))
    c = dict(config)
    results = _BENCHMARK(c, nepochs=num_epochs)
    train_acc = results[num_epochs]["train-acc"] / 100.0
    val_acc = results[num_epochs]["valid-acc"] / 100.0
    test_acc = results[num_epochs]["test-acc"] / 100.0

    return {
        "runtime": results[num_epochs]["runtime"],
        "train_regret": 1 - train_acc,
        "train_loss": 1 - train_acc,
        "val_regret": 1 - val_acc,
        "val_loss": 1 - val_acc,
        "test_regret": 1 - test_acc,
        "test_loss": 1 - test_acc,
    }


def run_dehbws(cs: Configuration, args: argparse.Namespace) -> Dict[str, Any]:
    """Run DEHB + weight sharing on a given configuration space
    Args:
        cs (Configuration): Configuration space
        args (Namespace): config to set up the dehb+weightsharing
            Note args.use_benchmark should be False for this setting

    Returns:
        dict: Final incumbent found by dehb+weightsharing
    """
    utils.seed_all(args.seed)
    create_supernet(args.seed, args.warm)
    ###########################
    # DEHB optimisation block #
    ###########################
    dimensions = len(cs.get_hyperparameters())
    args.output_path += "/" + str(args.seed)
    objective_fn = functools.partial(
        dehb_objective_fn,
        use_benchmark=args.use_benchmark,
        device=args.device,
        seed=args.seed,
        effective_budget=None,
    )
    dehb = DEHBWS(
        f=objective_fn,
        cs=cs,
        dimensions=dimensions,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        eta=args.eta,
        output_path=args.output_path,
        n_workers=args.n_workers,
        update_func=update_func,
    )

    traj, runtime, history = dehb.run(
        fevals=args.func_evals, verbose=args.verbose, seed=args.seed, device=args.device
    )
    # end of DEHB optimisation
    # Saving optimisation trace history

    name = time.strftime("%x %X %Z", time.localtime(dehb.start))
    name = name.replace("/", "-").replace(":", "-").replace(" ", "_")
    logger.info("Saving trajectory ...")
    with open(
        os.path.join(args.output_path, "trajectory_{}.pkl".format(name)), "wb"
    ) as f:
        pickle.dump(traj, f)
    incumbent = dehb.vector_to_configspace(dehb.inc_config)

    logger.info("Saving supernet model ...")
    torch.save(
        supernet.Supernet.state_dict(), args.output_path + "/supernet_model.state"
    )
    return incumbent


def run_dehb(cs: Configuration, args: argparse.Namespace) -> Dict:
    """Run dehb on a given configuration space

    Args:
        cs (Configuration): Configuration space
        args (Namespace): config to set up the dehb
            Note args.use_benchmark should be True for this setting

    Returns:
        dict: Final incumbent found by dehb
    """
    utils.seed_all(args.seed)
    ###########################
    # DEHB optimisation block #
    ###########################
    dimensions = len(cs.get_hyperparameters())
    args.output_path += "/" + str(args.seed)
    objective_fn = functools.partial(
        dehb_objective_fn,
        use_benchmark=args.use_benchmark,
        device=args.device,
        seed=args.seed,
    )
    dehb = DEHB(
        f=objective_fn,
        cs=cs,
        dimensions=dimensions,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        eta=args.eta,
        output_path=args.output_path,
        n_workers=args.n_workers,
    )
    traj, runtime, history = dehb.run(
        fevals=args.func_evals, verbose=args.verbose, seed=args.seed, device=args.device
    )
    # end of DEHB optimisation
    # Saving optimisation trace history

    name = time.strftime("%x %X %Z", time.localtime(dehb.start))
    name = name.replace("/", "-").replace(":", "-").replace(" ", "_")
    logger.info("Saving trajectory ...")
    with open(
        os.path.join(args.output_path, "trajectory_{}.pkl".format(name)), "wb"
    ) as f:
        pickle.dump(traj, f)
    incumbent = dehb.vector_to_configspace(dehb.inc_config)
    return incumbent


def run_smac(cs: Configuration, args: Dict) -> Dict:
    """Run SMAC on a given configuration space

    Args:
        cs (Configuration): Configuration space
        args (Namespace): config to set up the SMAC
            Note args.use_benchmark should be True for this setting

    Returns:
        dict: Final incumbent found by dehb
    """
    utils.seed_all(args.seed)
    args.output_path += "/" + str(args.seed)
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "ta_run_limit": args.func_evals,  # max duration to run the optimization (in seconds)
            "cs": cs,  # configuration space
            "output-dir": args.output_path,
            # working directory where intermediate results are stored
            "deterministic": True,
            "num_workers": args.n_workers
            # "limit_resources": True,  # Uses pynisher to limit memory and runtime
            # Then you should handle runtime and memory yourself in the TA
            # If you train the model on a CUDA machine, then you need to disable this option
            # "memory_limit": 8192,  # adapt this to reasonable value for your hardware
        }
    )
    # intensifier parameters (Budget parameters for BOHB)
    intensifier_kwargs = {
        "initial_budget": args.min_budget,
        "max_budget": args.max_budget,
        "eta": args.eta,
    }
    # To optimize, we pass the function to the SMAC-object
    objective_fn = functools.partial(
        smac_objective_fn,
        use_benchmark=args.use_benchmark,
        device=args.device,
        seed=args.seed,
        output_path=args.output_path,
    )
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(args.seed),
        tae_runner=objective_fn,
        intensifier_kwargs=intensifier_kwargs,
        # all arguments related to intensifier can be passed like this
        initial_design_kwargs={
            "n_configs_x_params": 1,  # how many initial configs to sample per parameter
            "max_config_fracs": 0.2,
        },
    )
    args.output_path = smac.output_dir
    # def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(),
    #                                       instance='1', budget=max_iters, seed=0)[1]
    # Start optimization
    try:  # try finally used to catch any interupt
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(
        config=incumbent, instance="1", budget=args.max_budget, seed=args.seed
    )[1]
    print("Optimized Value: %.4f" % inc_value)

    # srore your optimal configuration to disk
    opt_config = incumbent.get_dictionary()
    with open(args.output_path + "/opt_cfg.json", "w") as f:
        json.dump(opt_config, f)
    return incumbent


def main(args):
    utils.seed_all(args.seed)
    # Get configuration space
    cs = get_configspace(args.seed)
    run_nas_hpo = {"dehb": run_dehb, "smac": run_smac, "dehbws": run_dehbws}[args.type]
    incumbent = run_nas_hpo(cs, args)

    # Retrain and evaluate best found configuration
    if args.refit_training:
        logger.info("Retraining on complete training data to compute test metrics...")
        acc = train_and_evaluate(
            incumbent,
            args.retrain_budget,
            seed=args.seed,
            device=args.device,
            use_benchmark=args.use_test_benchmark,
        )
        logger.info(f"Test accuracy of {acc:.3f} for the best found configuration: ")
        logger.info(incumbent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = utils.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))

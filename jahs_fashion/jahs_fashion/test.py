import functools
import glob

import supernet
from run_hpo_nas import train_and_evaluate, objective_fn, dehb_objective_fn
from configspace import get_configspace
from supernet import create_supernet, update_func

create_supernet()
cs = get_configspace(42)

job_info = {
    "config": cs.sample_configuration(),
    "budget": 4,
    "kwargs": {"effective_budget": 2},
}

config, budget = job_info["config"], job_info["budget"]
kwargs = job_info["kwargs"]

objective_fn = functools.partial(
    dehb_objective_fn,
    use_benchmark=False,
    device="cuda",
    seed=42,
)

print(objective_fn(config, budget, **kwargs))

for i in range(2):
    config = cs.sample_configuration()
    print(config)
    print(objective_fn(config, budget=1, seed=42, device="cuda", use_benchmark=False))

files = glob.glob("tmp/*")
print(files)
cs = get_configspace(42)
configs = []

for i in range(10):
    config = cs.sample_configuration()
    config = dict(config)
    # Fixing constant config values here as there's a bug in DEHB
    config["N"] = 5
    config["W"] = 16
    config["Resolution"] = 1.0
    configs.append(config)

update_func(configs)

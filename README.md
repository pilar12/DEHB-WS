# AutoML lecture 2022 (Albert Ludwigs University of Freiburg)

## DEHB-WS: Joint Architecture and Hyperparameter Search with Weight Sharing

### Zahra Padar, Sharat Patil, Sai Prasanna

Python version: 3.8

## Install requirements:
```commandline
pip install -r requirements.txt
```
## Run DEHB-WS pipeline
### Warmstart Supernet:

```commandline
python3 warm_train_supernet.py
```

### Run DEHB-WS:
```commandline
python3 run_hpo_nas.py --configs defaults dehbws --seed 13
```
Logs and trained model are stored at: results/dehbws_results/{seed}


### Finetune and evaluate incumbent:
```commandline
python3 finetune.py
```
Tuned models are saved at: results/dehbws_results/{seed}

## Run DEHB or SMAC4MF
```commandline
python3 run_hpo_nas.py --configs defaults {dehb/smac} --seed 13
```

### Files
    
* configs.yaml : Default run configurations. Can be overridden by passing args

* configspace.py : JAHS-bench-201 config space 

* dehbws.py : DEHB-WS implementation.

* finetune.py : Script to finetune and evaluate DEHB-WS incumbent

* run_hpo_nas.py : Main script to perform Joint Architecture and Hyperparameter Search

* supernet.py : Global supernet creation and update function

* train.py : Script to train and score subnets

* utils.py : Utility functions

* warm_train_supernet.py : Script to warm start the supernet
* datasets/ : 8, 16 , 32 resolution Fashion-MNIST with fixed train-validation splits

* model/ : Code for Supernet
  * dynamic_model.py : JAHS-201 Supernet
  * dynamic_ops.py : Dynamic Layers
  * dynamic_primitives.py : Dynamic Architecture Blocks (Resnet, Cell)
* results/ : Directory to log and store runs
* utility_scripts: Scripts to simulate Hyper band runs


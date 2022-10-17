# AutoML lecture 2022 (Freiburg & Hannover)
## Final Project

This repository contains all things needed for the project 'jahs_fashion'.
Your task is to optimize a networks accuracy (maximize) on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset within 6 hours.
such that your network reaches a maximal accuracy.
The constraint values that we provide in the script are only an example. 

## Repo structure
*  Source folder      
    * [example_bohb.py](example_bohb.py) <BR>
      contains a simple example of how to use BOHB in SMAC on the dataset. 

    * [baseline.py](baseline.py)<BR>
      contains a simple baseline that reaches 0.902 accuracy on the test sets. The optimized network should be at least on par with this baseline. Further benchmarks can be found under [Fashion-MNIST Benchmark](https://github.com/zalandoresearch/fashion-mnist#benchmark).
    
    * [cnn.py](cnn.py)<BR>
      contains the source code of an example network you could optimize.
    
    * [utils.py](utils.py)<BR>
      contains simple helper functions for cnn.py

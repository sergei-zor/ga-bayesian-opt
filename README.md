# Bayesian Optimization of Genetic Algorithm Hyperparameters

This repository contains ongoing work for optimizing the hyperparameters (hps) of a genetic algorithm (GA) with Bayesian optimization (BO). The project is a continuation of [our previous work](https://doi.org/10.1016/j.commatsci.2025.114332) on genetic algorithm and active learning optimization of 3D lattice materials, extending it by systematically tuning GA hyperparameters to improve convergence, robustness, and final solution quality.

## Introduction

Genetic algorithms proven to be effetive for materials and topology optimization, but their performance depends strongly on hyperparameters such as: population size, mutation rate and others. At the same time, genetic algorithm is costly, it might take up to several days to converge when coupled with FFT mechanical simulations. Rather than selecting hyperparameters with random/grid search, this project uses Bayesian optimization to iteratively search the GA hps space. Each candidate hyperparameter set is evaluated by running a GA to obtain the highest achieved specific elastic modulus.To make this optimization computationally cheaper, elasticity evaluations are performed using a pretrained surrogate model, replacing expensive FFT-based mechanical simulations used in our previous work.

## General workflow

The optimization pipeline consists of two main parts:

1. **Bayesian optimization**

* Searches the GA hyperparameter space
* Uses Sobol sampling for initial design points (loaded from CSV if provided, generated otherwise)
* Supports multiple acquisition functions

2. **Genetic algorithm evaluations**

* Runs a shortened GA without active learning
* Evaluates lattice performance using a fixed surrogate model
* Returns scalar performance value to the BO loop

At each BO iteration:

* A new GA hyperparameter set is proposed
* The GA is executed with `ga_runner.py`
* Performance metrics are returned and used to update the BO model


## Running the code 

### Bayesian optimization

The Bayesian optimization is run with:

```
python bayesian_optimization.py --n_init 30 --ga_iter 25 --bo_iter 30 --acq ucb

```

Available arguments are:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--n_init` | `int` | `20` | Number of initial Sobol samples. |
| `--bo_iter` | `int` | `30` | Number of BO iterations. |
| `--ga_iter` | `int` | `25` | Number of GA iterations in each evaluation. |
| `--repeats_init` | `int` | `1` | Number of repeats per initial sample to average results. |
| `--repeats_acq` | `int` | `1` | Number of repeats per BO-acquired candidate. |
| `--acq` | `str` | `qnei` | Acquisition function: `ucb`, `nei`, `qnei`, `lognei`. |
| `--q` | `int` | `3` | Batch size (specific to `qNEI`). |
| `--objective` | `str` | `base` | Objective type for the GP: `base` or `penalized`. |
| `--save_csv` | `str` | `bo_evals.csv` | CSV file to save/append evaluation results. |
| `--load_prev` | `flag` | `False` | Use this flag to load previous Sobol samples from the csv. |

Initial Sobol samples are loaded from a .csv file if provided. otherwise, they are generated automatically.

### Genetic algorithm

The GA can also be executed separately for testing/debugging:

```
python ga_runner.py 

```

### Contributors

* Sergei Zorkaltsev
* Christina Schenk

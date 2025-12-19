# Bayesian Optimization of Genetic Algorithm Hyperparameters

*(Work in progress)*

This repository contains ongoing work for optimizing the hyperparameters (hps) of a genetic algorithm (GA) with Bayesian optimization (BO). The project is a continuation of [our previous work](https://doi.org/10.1016/j.commatsci.2025.114332) on genetic algorithm and active learning optimization of 3D lattice materials, extending it by systematically tuning GA hyperparameters to improve convergence, robustness, and final solution quality.

## Project motivation

Genetic algorithms proven to be effetive for materials and topology optimization, but their performance depends strongly on hyperparameters such as: population size, mutation rate and others. At the same time, genetic algorithm is costly, it might take up to several days to converge when coupled with FFT mechanical simulations. Rather than selecting hyperparameters with random/grid search, this project uses Bayesian optimization to iteratively search the GA hps space. Each candidate hyperparameter set is evaluated by running a GA to obtain the highest achieved specific elastic modulus.To make this optimization computationally cheaper, elasticity evaluations are performed using a pretrained surrogate model, replacing expensive FFT-based mechanical simulations used in our previous work.

## General workflow

The optimization pipeline consists of two main parts:

1. **Bayesian optimization**

* Searches the GA hyperparameter space
* Uses Sobol sampling for initial design points (loaded from CSV if provided, generated otherwise)
* Supports multiple acquisition functions

2. **Genetic algorithm evaluations**

* Runs a shortened GA without active learning
* Evaluates lattice structures using a fixed surrogate model
* Returns scalar performance metrics to the BO loop

At each BO iteration:

* A new GA hyperparameter set is proposed
* The GA is executed with `ga_runner.py`
* Performance metrics are returned and used to update the BO model


## Running the code 

### Bayesian optimization

The main entry point for Bayesian optimization is:

```
python bayes_opt/run_bo.py --acq ei

```
Supported acquisition functions are UCB, NEI, qNEI, logNEI.

Initial Sobol samples are loaded from a CSV file if provided; otherwise, they are generated automatically.

### Genetic algorithm

The GA can also be executed independently for testing or debugging:

```
python ga/ga_runner.py --smoke_test

```

### Contributors

* Sergei Zorkaltsev
* Christina Schenk

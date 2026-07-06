#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import csv
import time
from typing import Tuple, List
from tqdm import tqdm
import math
import random

import torch
from torch.quasirandom import SobolEngine

import botorch
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.utils.gpytorch_modules import get_matern_kernel_with_gamma_prior
from botorch.optim import optimize_acqf

try:
    from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
except Exception:
    from botorch.fit import fit_gpytorch_model

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize

import ga_runner

dtype = torch.double
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

log_tol = 1e-8

MUT_MIN = 0.0
MUT_MAX = 0.75
PARENTS_MIN = 10
PARENTS_MAX = 200
OFFSPR_MUT_MIN = 0.0
OFFSPR_MUT_MAX = 1.0

def decode_n_parents(norm_val):
    return int(round(norm_val * (PARENTS_MAX - PARENTS_MIN) + PARENTS_MIN))

def decode_mutation_rate(norm_val):
    return float(norm_val * (MUT_MAX - MUT_MIN) + MUT_MIN)

def decode_offspring_mut_frac(norm_val):
    return float(norm_val * (OFFSPR_MUT_MAX - OFFSPR_MUT_MIN) + OFFSPR_MUT_MIN)

import inspect
try:
    from botorch.sampling.samplers import SobolQMCNormalSampler
except Exception:
    try:
        from botorch.sampling.normal import SobolQMCNormalSampler
    except Exception:
        SobolQMCNormalSampler = None

def make_sobol_qmc_normal_sampler(n_samples=256):
    if SobolQMCNormalSampler is None:
        return None
    try:
        return SobolQMCNormalSampler(num_samples=n_samples)
    except TypeError:
        try:
            return SobolQMCNormalSampler(sample_shape=torch.Size([n_samples]))
        except Exception:
            try:
                return SobolQMCNormalSampler(n_samples)
            except Exception as e:
                sig = ''
                try:
                    sig = str(inspect.signature(SobolQMCNormalSampler.__init__))
                except Exception:
                    pass
                raise RuntimeError(
                    'Could not construct SobolQMCNormalSampler. '
                    f'Constructor signature: {sig}. Error: {e}'
                ) from e

def save_gp_state(model, train_X, train_Y_used, iteration=None, state_dir='gp_states'):
    os.makedirs(state_dir, exist_ok=True)
    ts = int(time.time())
    iter_str = f'iter{iteration}' if iteration is not None else 'init'
    fname = f'gp_{iter_str}_{ts}.pt'
    path = os.path.join(state_dir, fname)
    full_state = {'state_dict': model.state_dict(),
              'train_X': train_X.detach().cpu(),
              'train_Y_used': train_Y_used.detach().cpu()}
    try:
        torch.save(full_state, path)
        print(f'Saved GP state to {path}')
    except Exception as e:
        print(f'Error saving GP state: {e}')
    return path

def load_initial_data(csv_path):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    if not {'norm_mut','norm_par_frac','norm_offspr_mut','best_value'}.issubset(df.columns):
        return None, None
    X = df[['norm_mut', 'norm_par_frac', 'norm_offspr_mut']].values
    Y_raw = df[['best_value']].values  
    return torch.tensor(X, dtype=dtype, device=device), torch.tensor(Y_raw, dtype=dtype, device=device)

def save_rows(X_batch, Y_batch_raw, save_csv):
    '''
    Saves a candidate BO point  and corresponding reached value to a csv
    '''
    Xb = X_batch.cpu().numpy()
    Yb = Y_batch_raw.cpu().numpy()
    rows = []
    for i in range(Xb.shape[0]):
        norm_mut, norm_par_frac, norm_offspr_mut = map(float, Xb[i])
        mutation_rate = decode_mutation_rate(norm_mut)
        n_parents = decode_n_parents(norm_par_frac)
        offspr_mut = decode_offspring_mut_frac(norm_offspr_mut)
        best_val = float(Yb[i,0])
        best_val_norm = float(best_val / (1 + 0.15 * norm_par_frac))
        rows.append([time.time(), norm_mut, norm_par_frac, norm_offspr_mut,
                     mutation_rate, n_parents, offspr_mut, best_val, best_val_norm])
    with open(save_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def evaluate_hp_batch(X_norm, repeats=1, n_generations=25, iteration=0):
    '''
    Runs a GA with specified hyperparameters suggested by acq funtion
    Returns the tensor with best reached value of specific elastic modulus.
    '''
    assert X_norm.ndim == 2 and X_norm.shape[1] == 3
    results = []
    for row_idx in range(X_norm.shape[0]):
        row = X_norm[row_idx].to(device=device, dtype=dtype)
        norm_mut = float(row[0].item())
        norm_par_frac = float(row[1].item())
        norm_offspr_mut = float(row[2].item())

        mutation_rate = decode_mutation_rate(norm_mut)
        n_parents = decode_n_parents(norm_par_frac)
        offspring_mut_frac = decode_offspring_mut_frac(norm_offspr_mut)

        vals = []
        for r in range(repeats):
            run_idx = row_idx if iteration == 0 else r
            out = ga_runner.run_short_ga(n_generations=n_generations,
                                         mutation_rate=mutation_rate,
                                         n_parents=n_parents,
                                         offspring_mut_frac=offspring_mut_frac,
                                         iteration=iteration,
                                         run=run_idx,
                                         k=1)
            val = float(out)
            vals.append(val)
        mean_val = sum(vals) / len(vals)
        results.append([mean_val])
    res = torch.tensor(results, dtype=dtype, device=device)
    return res

def make_acqf(acq_name, model, train_X, sampler, q):
    '''
    Build acquisition function and return:
      acq_function, q_batch, description
    '''
    name = acq_name.lower()
    if name == 'ucb':
        return (UpperConfidenceBound(model=model, beta=1.0), 1, 'UCB(beta=1.0)')

    if name == 'nei':
        acq = qNoisyExpectedImprovement(model=model, X_baseline=train_X, sampler=sampler)
        return acq, 1, 'NEI (q=1)'

    if name == 'qnei':
        q_eff = max(1, int(q))
        acq = qNoisyExpectedImprovement(model=model, X_baseline=train_X, sampler=sampler)
        return acq, q_eff, f'qNEI (q={q_eff})'

    if name == 'lognei':
        acq = qNoisyExpectedImprovement(model=model, X_baseline=train_X, sampler=sampler)
        return acq, 1, 'logNEI'

    raise ValueError(f'Acquisition function {acq_name} is not supported')

def compute_used_targets(Y_raw, train_X, objective_choice, penalty = 0.15):
    '''
    Computes targets used in GP
    
    raw: y = E_max

    penalized: y = E_max / (1 + penalty * n_parents)
    '''
    if objective_choice == "raw":
        return Y_raw.clone()

    elif objective_choice == "penalized":
        n_par_norm = train_X[:, 1].unsqueeze(-1)
        denom = 1.0 + penalty * n_par_norm
        return (Y_raw / denom).to(dtype=Y_raw.dtype)

    else:
        raise ValueError(f"Objective choice '{objective_choice}' is not supported")

def run_bo(n_init=20,
           bo_iterations=30,
           ga_iterations=25,
           sobol_seed=0,
           repeats_init=1,
           repeats_acq=2,
           save_csv='bo_evals.csv',
           load_prev=False,
           acq_name='qnei',
           q_acq=3,
           objective='base'):
    
    torch.manual_seed(sobol_seed)
    random.seed(sobol_seed)

    bounds = torch.tensor([[0.0]*3, [1.0]*3], dtype=dtype, device=device)

    if load_prev:
        loaded_X, loaded_Y_raw = load_initial_data(save_csv)
    else:
        loaded_X, loaded_Y_raw = None, None

    if loaded_X is not None and loaded_Y_raw is not None:
        print(f'Loaded {loaded_X.shape[0]} initial samples from {save_csv}')
        train_X = loaded_X.to(dtype=dtype, device=device)
        train_Y_raw = loaded_Y_raw.to(dtype=dtype, device=device) 
        just_generated_initial = False
        csv_needs_header = False
    else:
        sobol = SobolEngine(dimension=3, scramble=True, seed=sobol_seed)
        X_init = sobol.draw(n_init).to(dtype=dtype, device=device)
        Y_init = evaluate_hp_batch(X_init, repeats=repeats_init, n_generations=ga_iterations, iteration=0)  # raw
        train_X = X_init
        train_Y_raw = Y_init
        just_generated_initial = True
        csv_needs_header = True

    if csv_needs_header:
        with open(save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'norm_mut', 'norm_par_frac', 'norm_offspr_mut',
                             'mutation_rate', 'n_parents', 'offspring_mut_frac', 'best_value', 'best_value_norm'])
    else:
        if not os.path.exists(save_csv):
            with open(save_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'norm_mut', 'norm_par_frac', 'norm_offspr_mut',
                                 'mutation_rate', 'n_parents', 'offspring_mut_frac', 'best_value', 'best_value_norm'])

    if just_generated_initial:
        save_rows(train_X, train_Y_raw, save_csv)

    train_Y_used = compute_used_targets(train_Y_raw, train_X, objective)
    is_log_acq = (acq_name.lower() == 'lognei')

    def make_train_targets_for_gp(Y_used, log_acq):
        if not log_acq:
            out = Y_used.clone()
            return out.to(dtype=dtype, device=device)
            
        min_val = float(Y_used.min().item())
        shift = max(0.0, log_tol - min_val)
        Y_shifted = Y_used + shift
        return torch.log(Y_shifted.to(dtype=dtype, device=device))

    train_Y_for_gp = make_train_targets_for_gp(train_Y_used, is_log_acq)
    dim = train_X.shape[-1]
    matern = get_matern_kernel_with_gamma_prior(ard_num_dims=dim)

    model = SingleTaskGP(train_X, train_Y_for_gp, covar_module=matern)   
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    save_gp_state(model, train_X, train_Y_for_gp, iteration=0)

    print(f'BO with acquisition {acq_name} and {objective} objective')

    for it in range(bo_iterations):
        print(f'\nBO iteration {it+1}/{bo_iterations}, train size = {train_X.shape[0]}')

        dim = train_X.shape[-1]
        matern = get_matern_kernel_with_gamma_prior(ard_num_dims=dim)
        model = SingleTaskGP(train_X,
                             train_Y_for_gp,   
                             covar_module=matern,
                             outcome_transform=Standardize(m=1)).to(train_X.device)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        model.eval() 

        sampler = None
        try:
            sampler = make_sobol_qmc_normal_sampler(n_samples=256)
        except Exception:
            print('Sampler creation failed')
            sampler = None

        d = train_X.shape[-1]
        bounds = torch.stack([
            torch.zeros(d, dtype=train_X.dtype, device=train_X.device),
            torch.ones(d, dtype=train_X.dtype, device=train_X.device)])

        acq_obj, chosen_q, acq_msg = make_acqf(acq_name, model, train_X, sampler, q_acq)
        print('Acquisition:', acq_msg)

        candidate, acq_val = optimize_acqf(acq_function=acq_obj,
                                           bounds=bounds,
                                           q=chosen_q,
                                           num_restarts=8,
                                           raw_samples=512)
        candidate = candidate.to(dtype=dtype, device=device)
        print('Candidate:', candidate)

        new_y_raw = evaluate_hp_batch(candidate, repeats=repeats_acq, n_generations=ga_iterations, iteration=it+1) 
        new_y_used = []
        c_np = candidate.cpu().numpy()
        new_y_np = new_y_raw.cpu().numpy().reshape(-1)
        for idx in range(candidate.shape[0]):
            norm_par_frac = float(c_np[idx,1])
            raw_val = float(new_y_np[idx])
            if objective == 'penalized':
                val_used = raw_val / (1 + 0.15 * norm_par_frac)
            else:
                val_used = raw_val
            new_y_used.append([val_used])
        new_y_used = torch.tensor(new_y_used, dtype=dtype, device=device)

        print('Observed raw:', new_y_raw.cpu().numpy(), ', used in GP:', new_y_used.cpu().numpy()) 

        train_X = torch.cat([train_X, candidate.to(dtype=dtype, device=device)], dim=0)
        train_Y_raw = torch.cat([train_Y_raw, new_y_raw.to(dtype=dtype, device=device)], dim=0) 
        train_Y_used = compute_used_targets(train_Y_raw, train_X, objective)
        train_Y_for_gp = make_train_targets_for_gp(train_Y_used, is_log_acq)

        save_rows(candidate, new_y_raw, save_csv)

        model.set_train_data(train_X, train_Y_for_gp, strict=False)
        fit_gpytorch_model(mll)
        save_gp_state(model, train_X, train_Y_for_gp, iteration=it+1)

    if objective == 'base':
        best_idx = int(torch.argmax(train_Y_raw))
        best_val_report = float(train_Y_raw[best_idx].item())
    else:
        best_idx = int(torch.argmax(train_Y_used))
        best_val_report = float(train_Y_raw[best_idx].item())  
        best_val_used = float(train_Y_used[best_idx].item())

    best_norm = train_X[best_idx].cpu().tolist()
    best_mut = decode_mutation_rate(best_norm[0])
    best_n_parents = decode_n_parents(best_norm[1])
    best_offspr_mut = decode_offspring_mut_frac(best_norm[2])

    print('\n=== BO finished ===')
    if objective == 'base':
        print(f'Best observed (base) objective: {best_val_report:.6f}')
    else:
        print(f'Best observed (used) objective ({objective}): {best_val_used:.6f} with corresponding raw value: {best_val_report:.6f}')
    print('Decoded best HPs:')
    print(f'mutation_rate = {best_mut:.6f}')
    print(f'n_parents = {best_n_parents}')
    print(f'offspring_mut_frac = {best_offspr_mut:.3f}')
    print(f'Normalized best point in [0,1]: {best_norm}')

    return {'train_X': train_X,
            'train_Y_raw': train_Y_raw,
            'train_Y_used': train_Y_used,
            'best_norm': best_norm,
            'best_value_raw': best_val_report,
            'best_decoded': {'mutation_rate': best_mut, 'n_parents': best_n_parents, 'offspring_mut_frac': best_offspr_mut}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BO for GA hyperparameters')
    parser.add_argument('--n_init', type=int, default=20, help='Number of Sobol initial samples')
    parser.add_argument('--bo_iter', type=int, default=30, help='Number of BO iterations')
    parser.add_argument('--ga_iter', type=int, default=25, help='Number of GA iterations')
    parser.add_argument('--repeats_init', type=int, default=1, help='Repeats per initial sample')
    parser.add_argument('--repeats_acq', type=int, default=1, help='Repeats per BO-acquired candidate')
    parser.add_argument('--save_csv', type=str, default='bo_evals.csv', help='CSV file to append evaluations')
    parser.add_argument('--load_prev', action='store_true', help='Load previous Sobol samples from CSV')
    parser.add_argument('--acq', type=str, default='qnei', choices=['ucb','nei','qnei','lognei'], help='Acquisition function')
    parser.add_argument('--q', type=int, default=3, help='q for qNEI (batch size)')
    parser.add_argument('--objective', type=str, default='base', choices=['base','penalized'], help='Objective used for GP (raw or normalized)')
    args = parser.parse_args()
    
    run_bo(n_init=args.n_init,
           bo_iterations=args.bo_iter,
           ga_iterations=args.ga_iter,
           repeats_init=args.repeats_init,
           repeats_acq=args.repeats_acq,
           save_csv=args.save_csv,
           load_prev=args.load_prev,
           acq_name=args.acq,
           q_acq=args.q,
           objective=args.objective)

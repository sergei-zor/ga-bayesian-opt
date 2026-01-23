#!/usr/bin/env python3
import argparse
import gpytorch
import os
import pandas as pd
import csv
import time
from typing import Tuple, List
from tqdm import tqdm
import torch
from torch.quasirandom import SobolEngine
import botorch
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
try:
    from botorch.acquisition.logei import LogNoisyExpectedImprovement
except Exception:
    from botorch.acquisition import LogNoisyExpectedImprovement
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

print("botorch", botorch.__version__)

# compatibility imports
import inspect
import numpy as np
try:
    from botorch.sampling.samplers import SobolQMCNormalSampler
except Exception:
    try:
        from botorch.sampling.normal import SobolQMCNormalSampler
    except Exception:
        SobolQMCNormalSampler = None

# fit helper
try:
    from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
except Exception:
    from botorch.fit import fit_gpytorch_model

# FixedNoiseGP optional import
try:
    from botorch.models.gp_regression import FixedNoiseGP
except Exception:
    try:
        from botorch.models import FixedNoiseGP as FixedNoiseGP
    except Exception:
        FixedNoiseGP = None

# exception type for detection
from botorch.exceptions.errors import BotorchTensorDimensionError

def make_sobol_qmc_normal_sampler(n_samples=256):
    if SobolQMCNormalSampler is None:
        raise RuntimeError(
            "SobolQMCNormalSampler not available in botorch installation. "
            "Please upgrade botorch or install a compatible version."
        )
    try:
        return SobolQMCNormalSampler(num_samples=n_samples)
    except TypeError:
        pass
    try:
        return SobolQMCNormalSampler(sample_shape=torch.Size([n_samples]))
    except TypeError:
        pass
    try:
        return SobolQMCNormalSampler(n_samples)
    except Exception as e:
        sig = ""
        try:
            sig = str(inspect.signature(SobolQMCNormalSampler.__init__))
        except Exception:
            pass
        raise RuntimeError(
            "Could not construct SobolQMCNormalSampler with common argument names. "
            f"Constructor signature: {sig!s}. Please upgrade botorch (`pip install -U botorch`)."
        ) from e

if FixedNoiseGP is None:
    print("FixedNoiseGP class not available in your installed botorch. ")

import ga_runner
import warnings
warnings.filterwarnings("ignore")

def save_gp_state(model, train_X, train_Y, iteration=None, state_dir="gp_states", prefix="gp"):
    os.makedirs(state_dir, exist_ok=True)
    ts = int(time.time())
    iter_str = f"iter{iteration}" if iteration is not None else "init"
    fname = f"{prefix}_{iter_str}_{ts}.pt"
    path = os.path.join(state_dir, fname)

    bundle = {
        "state_dict": model.state_dict(),
        "train_X": train_X.detach().cpu(),
        "train_Y": train_Y.detach().cpu(),
    }

    try:
        torch.save(bundle, path)
        print(f"[GP SAVE] Saved GP checkpoint → {path}")
    except Exception as e:
        print(f"[GP SAVE] ERROR saving checkpoint: {e}")

    return path

def make_noise_var(train_Y: torch.Tensor, sample_vars=None, repeats_per_obs: int = 1, dtype=torch.double, device=torch.device("cpu")):
    n = train_Y.shape[0]
    if sample_vars is not None:
        noise = torch.tensor(sample_vars, dtype=dtype, device=device) / float(repeats_per_obs)
        noise = noise.clamp_min(1e-12)
        return noise.unsqueeze(-1)  # shape [n,1]
    else:
        ystd = float(train_Y.squeeze(-1).std().cpu().item())
        if np.isnan(ystd) or ystd == 0.0:
            ystd = max(1.0, float(train_Y.squeeze(-1).abs().mean().cpu().item()))
        rel = 0.005
        noise_std = rel * ystd
        noise_var = (noise_std ** 2)
        noise_var_vec = torch.full((n, 1), noise_var, dtype=dtype, device=device)
        # ensure not zero
        return noise_var_vec.clamp_min(1e-12)

def load_initial_data_if_exists(csv_path):
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)

    X = df[["norm_mut", "norm_par_frac", "norm_offspr_mut"]].values
    X = torch.tensor(X, dtype=dtype, device=device)

    Y = df[["best_value"]].values
    Y = torch.tensor(Y, dtype=dtype, device=device)

    return X, Y

def save_rows(X_batch: torch.Tensor, Y_batch: torch.Tensor, save_csv):
    Xb = X_batch.cpu().numpy()
    Yb = Y_batch.cpu().numpy()
    rows = []
    for i in range(Xb.shape[0]):
        norm_mut, norm_par_frac, norm_offspr_mut = map(float, Xb[i])
        mutation_rate = decode_mutation_rate(norm_mut)
        n_parents = decode_n_parents(norm_par_frac)
        offspr_mut = decode_offspring_mut_frac(norm_offspr_mut)
        best_val = float(Yb[i,0])
        rows.append([time.time(), norm_mut, norm_par_frac, norm_offspr_mut,
                     mutation_rate, n_parents, offspr_mut, best_val])
    with open(save_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

dtype = torch.double
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MUT_MIN = 0.0
MUT_MAX = 0.75
PARENTS_MIN = 10
PARENTS_MAX = 175
OFFSPR_MUT_MIN = 0.0
OFFSPR_MUT_MAX = 1.0

def decode_n_parents(norm_val: float) -> int:
    return int(round(norm_val * (PARENTS_MAX - PARENTS_MIN) + PARENTS_MIN))

def decode_mutation_rate(norm_val: float) -> float:
    return float(norm_val * (MUT_MAX - MUT_MIN) + MUT_MIN)

def decode_offspring_mut_frac(norm_val: float) -> float:
    return float(norm_val * (OFFSPR_MUT_MAX - OFFSPR_MUT_MIN) + OFFSPR_MUT_MIN)


def evaluate_hp_batch(
    X_norm: torch.Tensor,
    repeats: int = 1,
    short_gen: int = 8,
    verbose: bool = False,
    iteration: int = 0
) -> torch.Tensor:

    assert X_norm.ndim == 2 and X_norm.shape[1] == 3
    results: List[List[float]] = []

    for row_idx in range(X_norm.shape[0]):
        row = X_norm[row_idx].to(device=device, dtype=dtype)
        norm_mut = float(row[0].item())
        norm_par_frac = float(row[1].item())
        norm_offspr_mut = float(row[2].item())

        mutation_rate = decode_mutation_rate(norm_mut)
        n_parents = decode_n_parents(norm_par_frac)
        offspring_mut_frac = decode_offspring_mut_frac(norm_offspr_mut)

        if verbose:
            print(f"Evaluating candidate {row_idx}: mut_rate={mutation_rate:.4f}, "
                  f"n_parents={n_parents}, offspring_mut_frac={offspring_mut_frac:.3f} (repeats={repeats})")

        vals = []
        for r in range(repeats):
            if iteration == 0:
                run_idx = row_idx
            else:
                run_idx = r
            out = ga_runner.run_short_ga(
                mutation_rate=mutation_rate,
                n_parents=n_parents,
                offspring_mut_frac=offspring_mut_frac,
                iteration = iteration,
                run = run_idx
            )
            if isinstance(out, dict):
                val = out.get("best_value", out.get("best", None) or out.get("best_f", None))
                if val is None:
                    raise RuntimeError("Returned dict but no 'best_value' key found.")
                val = float(val)
            else:
                val = float(out)
            vals.append(val)

        mean_val = sum(vals) / len(vals)
        results.append([mean_val])

    res = torch.tensor(results, dtype=dtype, device=device)  # produces shape [batch, 1]
    return res.view(-1, 1)

def run_bo(
    n_init: int = 20,
    bo_iterations: int = 30,
    sobol_seed: int = 0,
    repeats_init: int = 1,
    repeats_acq: int = 1,
    short_gen: int = 8,
    save_csv: str = "bo_evals.csv",
    load_prev: bool = False,
    q_acq: int = 1,
    sampler_n_samples: int = 256
):
    torch.manual_seed(sobol_seed)

    bounds = torch.tensor([[0.0]*3, [1.0]*3], dtype=dtype, device=device)

    if load_prev:
        loaded_X, loaded_Y = load_initial_data_if_exists(save_csv)
    else:
        loaded_X, loaded_Y = None, None

    if loaded_X is not None and loaded_Y is not None:
        train_X = loaded_X.to(dtype=dtype, device=device)
        train_Y = loaded_Y.to(dtype=dtype, device=device)  # shape [n,1]
        csv_needs_header = False
        just_generated_initial = False
        print(f"Loaded {train_X.shape[0]} samples from {save_csv}")

    else:
        sobol = SobolEngine(dimension=3, scramble=True, seed=sobol_seed)
        X_init = sobol.draw(n_init).to(dtype=dtype, device=device)
        Y_init = evaluate_hp_batch(X_init, repeats=repeats_init, short_gen=short_gen, iteration=0)
        assert Y_init.ndim == 2 and Y_init.shape[1] == 1
        train_X = X_init
        train_Y = Y_init
        csv_needs_header = True
        just_generated_initial = True

    if train_Y.dim() == 1:
        train_Y = train_Y.unsqueeze(-1)

    if csv_needs_header:
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "norm_mut", "norm_par_frac", "norm_offspr_mut",
                             "mutation_rate", "n_parents", "offspring_mut_frac", "best_value"])
    else:
        if not os.path.exists(save_csv):
            with open(save_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "norm_mut", "norm_par_frac", "norm_offspr_mut",
                                 "mutation_rate", "n_parents", "offspring_mut_frac", "best_value"])

    if just_generated_initial:
        save_rows(train_X, train_Y, save_csv)

    # Prepare log-targets and attempt to construct GP with compatibility
    train_Y_log_2d = torch.log(train_Y.clamp_min(1e-8))         # shape [n,1]
    train_Y_log_1d = train_Y_log_2d.squeeze(-1)                # shape [n]

    use_1d_labels = False
    train_Y_log = None
    try:
        model = SingleTaskGP(train_X, train_Y_log_1d)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        use_1d_labels = True
        train_Y_log = train_Y_log_1d
        print("Constructed SingleTaskGP with 1-D targets (shape [n]).")
    except BotorchTensorDimensionError:
        # try 2-d targets
        model = SingleTaskGP(train_X, train_Y_log_2d)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        use_1d_labels = False
        train_Y_log = train_Y_log_2d
        print("Constructed SingleTaskGP with 2-D targets (shape [n,1]).")

    # Save initial GP checkpoint (keep raw train_Y for bookkeeping)
    save_gp_state(model, train_X, train_Y, iteration=0)

    # BO loop
    for it in tqdm(range(bo_iterations)):
        print(f"BO iteration {it+1}/{bo_iterations}  |  train size = {train_X.shape[0]}")

        # create sampler (compat across versions)
        try:
            sampler = make_sobol_qmc_normal_sampler(n_samples=sampler_n_samples)
        except RuntimeError as e:
            print(f"Warning: could not create SobolQMCNormalSampler: {e}. Falling back to analytic EI.")
            sampler = None

        # Prefer LogNoisyExpectedImprovement (log-NEI) for models trained on log(y).
        # We intentionally avoid qNoisyExpectedImprovement (q>1) here and target a single candidate per iteration.
        acq = None
        try:
            if sampler is None:
                # If we can't create the sampler, LogNoisyExpectedImprovement may still accept sampler=None in some versions,
                # but to be safe we treat missing sampler as a signal to skip noisy MC acquisitions.
                raise RuntimeError("No sampler available for noisy MC acquisition.")
            # Try constructing LogNoisyExpectedImprovement (Monte Carlo based) for log-targets
            acq = LogNoisyExpectedImprovement(model=model, X_baseline=train_X, sampler=sampler)
            print("Using LogNoisyExpectedImprovement (log-NEI).")
        except Exception as e:
            # If LogNEI fails (common with some botorch/gpytorch combinations), fall back to analytic EI
            print("Warning: LogNoisyExpectedImprovement unavailable or construction failed:", e)
            print("Falling back to analytic ExpectedImprovement (single-point).")
            # Use the best observed log-value as best_f consistent with GP training
            if use_1d_labels:
                best_f = float(train_Y_log.max().item())
            else:
                # train_Y_log is 2-D [n,1] -> squeeze to scalar
                best_f = float(train_Y_log.max().item())
            acq = ExpectedImprovement(model=model, best_f=best_f)

        # optimize acquisition (we force single candidate q=1 since we are not using qNEI)
        q = 1
        candidate, acq_val = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q,
            num_restarts=8,
            raw_samples=512
        )

        # Evaluate candidate(s) in original units
        new_y = evaluate_hp_batch(candidate, repeats=repeats_acq, short_gen=short_gen, verbose=True, iteration=it+1)
        candidate = candidate.to(dtype=dtype, device=device)
        new_y = new_y.to(dtype=dtype, device=device)   # shape [q,1]

        train_X = torch.cat([train_X, candidate], dim=0)            # [N+q, d]
        train_Y = torch.cat([train_Y, new_y], dim=0)                # [N+q, 1]

        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)

        save_rows(candidate, new_y, save_csv)

        # Build log labels for the new observations (both shapes)
        new_y_log_2d = torch.log(new_y.clamp_min(1e-8))   # [q,1]
        new_y_log_1d = new_y_log_2d.squeeze(-1)           # [q]

        if use_1d_labels:
            train_Y_log = torch.cat([train_Y_log, new_y_log_1d], dim=0)   # shape [N+q]
            targets_for_gp = train_Y_log
        else:
            train_Y_log = torch.cat([train_Y_log, new_y_log_2d], dim=0)   # shape [N+q,1]
            targets_for_gp = train_Y_log

        # update GP training data and recreate MLL before fitting
        model.set_train_data(inputs=train_X, targets=targets_for_gp, strict=False)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
# --- BEGIN: ensure GP prior-mean has same trailing shape as Y (match [n,1]) ---
# In some botorch/gpytorch combos the mean_module returns shape [...,] while
# your train targets are [...,1]; wrap the mean module so it returns a column.

        class _ColumnMeanWrapper(gpytorch.means.Mean):
            def __init__(self, base_mean_module: gpytorch.means.Mean):
                super().__init__()
                self.base = base_mean_module

            def forward(self, x):
                m = self.base(x)
                if m.ndim == x.ndim - 1 or m.ndim == 1:
                    return m.unsqueeze(-1)
                return m


        try:
            with torch.no_grad():
                test_x = train_X[:1]  # 1 x d
                prior_mean = model.mean_module(test_x)
                if prior_mean.ndim == train_Y_log.ndim - 1:
                    model.mean_module = _ColumnMeanWrapper(model.mean_module)
                    print("Wrapped model.mean_module to return column outputs (match Y shape [n,1]).")
        except Exception as _e:
            print("Warning (mean-wrapper): could not auto-test/wrap mean_module:", _e)
        fit_gpytorch_model(mll)

        # Save GP + data checkpoint for this iteration
        save_gp_state(model, train_X, train_Y, iteration=it+1)

    # Finished BO: pick best
    best_idx = int(torch.argmax(train_Y))
    best_norm = train_X[best_idx].cpu().tolist()
    best_val = float(train_Y[best_idx].item())
    best_mut = decode_mutation_rate(best_norm[0])
    best_n_parents = decode_n_parents(best_norm[1])
    best_offspr_mut = decode_offspring_mut_frac(best_norm[2])

    print("\n=== BO finished ===")
    print(f"Best observed (mean) objective: {best_val:.6f}")
    print("Decoded best HPs:")
    print(f"  mutation_rate = {best_mut:.6f}")
    print(f"  n_parents = {best_n_parents}")
    print(f"  offspring_mut_frac = {best_offspr_mut:.3f}")
    print(f"Normalized best point (in [0,1]^3): {best_norm}")

    return {
        "train_X": train_X,
        "train_Y": train_Y,
        "best_norm": best_norm,
        "best_value": best_val,
        "best_decoded": {"mutation_rate": best_mut,
                         "n_parents": best_n_parents, "offspring_mut_frac": best_offspr_mut},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BO for GA hyperparameters using runner function")
    parser.add_argument("--n_init", type=int, default=25, help="Number of Sobol initial samples")
    parser.add_argument("--bo_iterations", type=int, default=30, help="Number of BO iterations")
    parser.add_argument("--repeats_init", type=int, default=1, help="Repeats per initial sample (for noise reduction)")
    parser.add_argument("--repeats_acq", type=int, default=1, help="Repeats per BO-acquired candidate")
    parser.add_argument("--save_csv", type=str, default="bo_evals.csv", help="CSV file to append evaluations")
    parser.add_argument("--load_prev", action="store_true", help="Load previous Sobol samples")
    args = parser.parse_args()

    run_bo(
        n_init=args.n_init,
        bo_iterations=args.bo_iterations,
        repeats_init=args.repeats_init,
        repeats_acq=args.repeats_acq,
        save_csv=args.save_csv,
        load_prev=args.load_prev
    )



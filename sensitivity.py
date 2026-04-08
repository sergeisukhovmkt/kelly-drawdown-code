"""
sensitivity.py — λ* sensitivity sweep (Section 6.2a, Sukhov 2026).

Sweeps λ over a grid for each combination of (μ, σ, δ, T) and records
the optimal λ* = argmax U(π, λ).

Runtime: ~3 min on a modern laptop (120 cells × 10 λ values × 3,000 paths).
Use SENS_N = 1_000 in config.py for a quick run during development.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from functools import partial
from scipy.stats import linregress

import config as cfg
from simulation import simulate
from policies import exp_ddr


def lambda_sweep_single(
    mu:    float,
    sigma: float,
    delta: float,
    T:     float,
    lambda_grid: np.ndarray = cfg.SENS_LAMBDA,
    n_paths:     int        = cfg.SENS_N,
    seed:        int        = 7,
) -> tuple[float, float, float]:
    """
    Find λ* = argmax U(π, λ) for a single parameter combination.

    Returns (lambda_star, U_star, kappa).
    """
    b     = -np.log(1.0 - delta)
    kappa = (mu - cfg.R) / sigma**2

    best_lam, best_U = lambda_grid[0], -1e9

    for lam in lambda_grid:
        policy = partial(exp_ddr, lam=lam, kappa=kappa, b=b)
        res = simulate(
            policy,
            n_paths=n_paths,
            seed=seed,
            mu=mu,
            sigma=sigma,
            delta=delta,
            T=T,
            kappa=kappa,
        )
        if res["penalised_U"] > best_U:
            best_U  = res["penalised_U"]
            best_lam = lam

    return best_lam, best_U, kappa


def run_sensitivity(
    mu_grid:    list[float] = cfg.SENS_MU,
    sigma_grid: list[float] = cfg.SENS_SIGMA,
    delta_grid: list[float] = cfg.SENS_DELTA,
    T_grid:     list[float] = cfg.SENS_T,
    **kwargs,
) -> pd.DataFrame:
    """
    Run the full sensitivity sweep and return a tidy DataFrame.

    Columns: mu, sigma, delta, T, kappa, lambda_star, U_star
    """
    rows = []
    total = len(mu_grid) * len(sigma_grid) * len(delta_grid) * len(T_grid)
    count = 0

    for mu in mu_grid:
        for sigma in sigma_grid:
            for delta in delta_grid:
                for T in T_grid:
                    lam, U, kap = lambda_sweep_single(
                        mu, sigma, delta, T, **kwargs
                    )
                    rows.append({
                        "mu":         mu,
                        "sigma":      sigma,
                        "delta":      delta,
                        "T":          T,
                        "kappa":      round(kap, 3),
                        "lambda_star": round(lam, 3),
                        "U_star":     round(U, 4),
                    })
                    count += 1
                    print(f"  [{count:3d}/{total}] "
                          f"μ={mu:.2f} σ={sigma:.2f} δ={delta:.0%} T={T:.0f}  "
                          f"κ={kap:.2f}  λ*={lam:.3f}")

    return pd.DataFrame(rows)


def fit_log_linear(df: pd.DataFrame) -> dict:
    """
    Fit  λ* ≈ a + b·ln(κ)  across all cells.

    Returns dict with intercept, slope, R².
    Reproduces Remark 6.2 in the paper.
    """
    log_kappa = np.log(df["kappa"].values)
    lam_star  = df["lambda_star"].values

    slope, intercept, r, *_ = linregress(log_kappa, lam_star)

    return {
        "intercept": round(intercept, 3),
        "slope":     round(slope, 3),
        "R2":        round(r**2, 3),
        "formula":   f"λ* ≈ {intercept:.2f} + ({slope:.2f})·ln(κ)",
    }


if __name__ == "__main__":
    print("Running sensitivity sweep (120 cells) ...")
    df = run_sensitivity()
    df.to_csv("Table_sensitivity_lambda.csv", index=False)
    print("\nResults saved to Table_sensitivity_lambda.csv")

    fit = fit_log_linear(df)
    print(f"\nLog-linear fit: {fit['formula']}  (R² = {fit['R2']})")
    print("\nMarginal means:")
    for col in ["mu", "sigma", "delta", "T"]:
        print(f"  λ* by {col}:")
        print("   ", df.groupby(col)["lambda_star"].mean().round(3).to_dict())

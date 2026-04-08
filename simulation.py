"""
simulation.py — Moving-barrier Monte Carlo engine.

Implements the full running-maximum drawdown simulation used in
Section 6 of Sukhov (2026).

Design
------
* Fully vectorised over paths (no Python loops over paths).
* Reproducible via explicit numpy Generator seeding.
* Returns per-path (log_wealth, survived) arrays for downstream analysis.

Key formulas
------------
Log-wealth update (Itô discrete-time):
    log W_{t+Δt} = log W_t + [r + f(μ−r) − ½f²σ²]Δt + fσ√Δt · ε_t
    ε_t ~ iid N(0,1)

Running maximum:
    M_t = max_{s≤t} W_s

Effective Ruin at step t:
    W_t / M_t < 1 − δ   ⟺   d_t < 0

Distance to barrier:
    d_t = log(W_t / M_t) + b,   b = −log(1−δ)
"""

from __future__ import annotations

import numpy as np
from typing import Callable

import config as cfg


PolicyFn = Callable[[np.ndarray], np.ndarray]
"""Type alias: maps log-distance array (N,) → leverage array (N,)."""


def simulate(
    policy:       PolicyFn,
    n_paths:      int   = cfg.N_TEST,
    seed:         int   = cfg.SEED_TEST,
    mu:           float = cfg.MU,
    r:            float = cfg.R,
    sigma:        float = cfg.SIGMA,
    delta:        float = cfg.DELTA,
    T:            float = cfg.T_SIM,
    dt:           float = cfg.DT,
    ruin_penalty: float = cfg.RUIN_PENALTY,
    kappa:        float = cfg.KAPPA,
) -> dict:
    """
    Run moving-barrier Monte Carlo for a given leverage policy.

    Parameters
    ----------
    policy       : function d_t (N,) → f_t (N,), clipped to [0, κ] internally
    n_paths      : number of simulation paths
    seed         : RNG seed for reproducibility
    ruin_penalty : p in U(π) = E[log W_T · 1{τ>T}] + p · P(τ≤T)

    Returns
    -------
    dict with keys:
        log_wealth   : (N,) final log-wealth (0 for ruined paths)
        survived     : (N,) boolean survival indicator
        penalised_U  : scalar penalised utility
        survival_pct : scalar survival rate in %
        mean_f       : (N,) mean realised leverage over surviving path
    """
    b      = -np.log(1.0 - delta)
    n_steps = int(T / dt)

    rng   = np.random.default_rng(seed)
    W     = np.ones(n_paths)        # wealth (normalised to 1)
    M     = np.ones(n_paths)        # running maximum
    alive = np.ones(n_paths, bool)  # True = not yet ruined

    f_sum   = np.zeros(n_paths)     # cumulative leverage (for mean)
    f_count = np.zeros(n_paths)     # steps alive

    for _ in range(n_steps):
        # ── State variable ────────────────────────────────────────────────────
        dd = 1.0 - W / M                                    # drawdown fraction
        d  = np.clip(np.log(np.maximum(1.0 - dd, 1e-15)) + b, 0.0, b)

        # ── Policy evaluation ─────────────────────────────────────────────────
        f = np.clip(policy(d), 0.0, kappa)

        # ── Accumulate leverage stats ─────────────────────────────────────────
        f_sum[alive]   += f[alive]
        f_count[alive] += 1

        # ── Log-wealth update (Itô discretisation) ────────────────────────────
        eps  = rng.standard_normal(n_paths)
        dX   = (r + f * (mu - r) - 0.5 * f**2 * sigma**2) * dt \
               + f * sigma * np.sqrt(dt) * eps
        W[alive] *= np.exp(dX[alive])

        # ── Update running maximum ────────────────────────────────────────────
        M[alive] = np.maximum(M[alive], W[alive])

        # ── Check for ruin ────────────────────────────────────────────────────
        newly_ruined          = alive & ((1.0 - W / M) >= delta)
        alive[newly_ruined]   = False

    # ── Post-simulation statistics ────────────────────────────────────────────
    log_W        = np.where(alive, np.log(W), np.nan)
    penalised_U  = float(np.mean(
        np.where(alive, np.log(W), ruin_penalty)
    ))
    survival_pct = float(alive.mean() * 100.0)
    mean_f       = np.where(f_count > 0, f_sum / f_count, 0.0)

    return {
        "log_wealth":    log_W,
        "survived":      alive,
        "penalised_U":   penalised_U,
        "survival_pct":  survival_pct,
        "mean_f":        mean_f,
    }


def bootstrap_ci(
    log_wealth:  np.ndarray,
    survived:    np.ndarray,
    ruin_penalty: float = cfg.RUIN_PENALTY,
    n_boot:      int    = cfg.N_BOOTSTRAP,
    ci_level:    float  = cfg.CI_LEVEL,
    seed:        int    = 99,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for penalised utility U(π).

    Returns (lower, upper) at the requested CI level.
    """
    rng   = np.random.default_rng(seed)
    n     = len(log_wealth)
    u_vec = np.where(survived, log_wealth, ruin_penalty)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx            = rng.integers(0, n, size=n)
        boot_means[i]  = u_vec[idx].mean()

    alpha = 1.0 - ci_level
    lo    = float(np.percentile(boot_means, 100 * alpha / 2))
    hi    = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def run_comparison(
    policies:     dict[str, PolicyFn],
    n_paths:      int   = cfg.N_TEST,
    seed:         int   = cfg.SEED_TEST,
    T:            float = cfg.T_SIM,
    ruin_penalty: float = cfg.RUIN_PENALTY,
    **kwargs,
) -> list[dict]:
    """
    Run all policies and return a list of result dicts (one per strategy),
    matching the structure of Table 3 in the paper.
    """
    rows = []
    for name, policy in policies.items():
        res = simulate(policy, n_paths=n_paths, seed=seed, T=T,
                       ruin_penalty=ruin_penalty, **kwargs)

        lo, hi = bootstrap_ci(
            res["log_wealth"], res["survived"],
            ruin_penalty=ruin_penalty,
        )

        # Mean annualised log-return on surviving paths
        surv_lw = res["log_wealth"][res["survived"]]
        ann_ret = (float(np.mean(surv_lw)) / T * 100.0) if len(surv_lw) else 0.0

        rows.append({
            "Strategy":      name,
            "U(π)":          round(res["penalised_U"], 3),
            "Survival %":    round(res["survival_pct"], 1),
            "CI_lo":         round(lo, 3),
            "CI_hi":         round(hi, 3),
            "Ann log-ret %": round(ann_ret, 2),
            "Mean f":        round(float(np.nanmean(res["mean_f"]
                                     [res["survived"]])), 4),
        })
    return rows


if __name__ == "__main__":
    from policies import make_policies
    import pandas as pd

    policies = make_policies()
    print("Running Monte Carlo comparison (N=15,000, seed=42) ...")
    rows = run_comparison(policies)
    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))

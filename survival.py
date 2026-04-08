"""
survival.py — Flat-barrier survival probability and survival-optimal leverage.

Implements:
    * Theorem 3.1  — closed-form survival probability S(f, T, d0)
    * Conjecture 3.2 verification — numerically optimal survival leverage f*_surv
    * Closed-form discounted value function u(d) from Theorem 4.2

References
----------
Karatzas & Shreve (1991), Theorem 3.5.7
Grossman & Zhou (1993)
Sukhov (2026), Theorems 3.1, 4.2
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

import config as cfg


# ── Theorem 3.1: Survival probability (flat-barrier approximation) ─────────────

def survival_prob(
    f: float | np.ndarray,
    T: float,
    d0: float,
    mu: float = cfg.MU,
    r: float  = cfg.R,
    sigma: float = cfg.SIGMA,
) -> float | np.ndarray:
    """
    P(τ > T) under flat-barrier approximation with fixed leverage f.

    Uses the reflection-principle formula (Theorem 3.1):

        S(f,T,d0) = Φ((d0 + μ_f T) / (σ_f √T))
                  − exp(−2 μ_f d0 / σ_f²) · Φ((−d0 + μ_f T) / (σ_f √T))

    Parameters
    ----------
    f   : leverage (scalar or array)
    T   : horizon in years
    d0  : initial log-distance to barrier
    """
    f    = np.asarray(f, dtype=float)
    mu_f = r + f * (mu - r) - 0.5 * f**2 * sigma**2   # log-wealth drift
    sf   = f * sigma                                    # log-wealth diffusion

    # Handle f = 0 separately to avoid division by zero
    safe_sf = np.where(sf > 1e-12, sf, 1e-12)

    a = (d0 + mu_f * T) / (safe_sf * np.sqrt(T))
    b = (-d0 + mu_f * T) / (safe_sf * np.sqrt(T))

    with np.errstate(over="ignore", invalid="ignore"):
        exponent = -2.0 * mu_f * d0 / safe_sf**2
        result   = norm.cdf(a) - np.exp(np.clip(exponent, -500, 500)) * norm.cdf(b)

    return np.where(sf > 1e-12, np.clip(result, 0.0, 1.0), 1.0)


# ── Conjecture 3.2: Survival-optimal leverage ─────────────────────────────────

def survival_optimal_leverage(
    T: float,
    d0: float,
    kappa: float = cfg.KAPPA,
    mu: float    = cfg.MU,
    r: float     = cfg.R,
    sigma: float = cfg.SIGMA,
) -> tuple[float, float]:
    """
    f*_surv(d0, T) = argmax_{f≥0} S(f, T, d0).

    Returns (f_star, S_star).
    Uses scalar minimisation of −S on [0, 3κ].
    """
    obj = lambda f: -survival_prob(f, T, d0, mu, r, sigma)
    res = minimize_scalar(obj, bounds=(0.0, 3.0 * kappa), method="bounded")
    f_star = float(res.x)
    s_star = float(-res.fun)
    return f_star, s_star


def build_survival_table(
    T_grid:  list[float] | None = None,
    d0_grid: list[float] | None = None,
    kappa: float = cfg.KAPPA,
    b: float     = cfg.B,
    **kwargs,
) -> list[dict]:
    """
    Build Table 1 from the paper: survival-optimal leverage over (T, d0/b) grid.
    """
    if T_grid  is None: T_grid  = [0.25, 0.50, 1.00, 2.00]
    if d0_grid is None: d0_grid = [0.25, 0.50, 0.75, 1.00]  # in units of d0/b

    rows = []
    for T in T_grid:
        for d0b in d0_grid:
            d0 = d0b * b
            f_star, s_star = survival_optimal_leverage(T, d0, kappa=kappa, **kwargs)
            s_kelly = float(survival_prob(kappa, T, d0, **kwargs))
            rows.append({
                "T":      T,
                "d0/b":   d0b,
                "f*_surv": round(f_star, 4),
                "S(f*)":  round(s_star, 4),
                "S(κ)":   round(s_kelly, 4),
            })
    return rows


# ── Theorem 4.2: Discounted HJB closed-form solution ─────────────────────────

def alpha_discounted(r: float, kappa: float, sigma: float) -> float:
    """
    α = 2r / (2r + κ²σ²)  ∈ (0, 1)   (Theorem 4.2)
    """
    return 2.0 * r / (2.0 * r + kappa**2 * sigma**2)


def value_function_discounted(d: np.ndarray, b: float, alpha: float) -> np.ndarray:
    """
    u(d) = (d/b)^α    (Theorem 4.2, closed-form discounted value function)
    """
    return np.power(np.clip(d / b, 0.0, 1.0), alpha)


def leverage_discounted(
    d: np.ndarray,
    kappa: float,
    alpha: float,
    b: float,
) -> np.ndarray:
    """
    f*(d) = κ d / (1 − α)   (Theorem 4.2, exact optimal leverage)

    Peak leverage at d = b:  f*(b) = κb/(1−α)  ≠ κ for b < 1.
    As r → 0 (α → 0):        f*(d) → κd  (same normalised profile as GZ).
    """
    return kappa * d / (1.0 - alpha)


if __name__ == "__main__":
    import pandas as pd

    print("=" * 60)
    print("Table 1 — Survival-Optimal Leverage  (Conjecture 3.2)")
    print("=" * 60)
    rows = build_survival_table()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print()
    print("=" * 60)
    print("Theorem 4.2 — Discounted closed-form leverage")
    print(f"Parameters: r=0.01, κ={cfg.KAPPA}, σ={cfg.SIGMA}, b={cfg.B:.4f}")
    print("=" * 60)
    r_disc = 0.01
    alpha  = alpha_discounted(r_disc, cfg.KAPPA, cfg.SIGMA)
    print(f"  α = {alpha:.5f},  1−α = {1-alpha:.5f}")
    d_vals = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0]) * cfg.B
    f_vals = leverage_discounted(d_vals, cfg.KAPPA, alpha, cfg.B)
    u_vals = value_function_discounted(d_vals, cfg.B, alpha)
    print(f"  {'d/b':>6}  {'d':>8}  {'u(d)':>8}  {'f*(d)':>8}")
    for db, d, u, f in zip(d_vals / cfg.B, d_vals, u_vals, f_vals):
        print(f"  {db:6.2f}  {d:8.4f}  {u:8.4f}  {f:8.4f}")

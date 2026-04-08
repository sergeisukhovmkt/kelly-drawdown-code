"""
policies.py — Leverage rules studied in Sukhov (2026).

Each policy is a callable  d → f(d)  where d is the log-distance
to the running-maximum barrier and f is the fraction of wealth
invested in the risky asset.

All rules satisfy f(0) = 0 (full de-risk at the barrier) and
f(b) ≤ κ (never exceed unconstrained Kelly).

Rules
-----
full_kelly        f(d) = κ                    [fixed-fraction, no de-risking]
half_kelly        f(d) = κ/2                  [fixed-fraction]
linear_ddr        f(d) = κ · d/b              [Grossman-Zhou (1993) implied rule]
exp_ddr           f(d) = κ(1 − e^{−λd/b})    [Definition 4.5, Sukhov (2026)]
powerlaw_ddr      f(d) = κd/(1−α)             [Theorem 4.2, discounted optimal]
"""

from __future__ import annotations

import numpy as np
from functools import partial

import config as cfg
from survival import alpha_discounted


# ── Individual policy functions ────────────────────────────────────────────────

def full_kelly(d: np.ndarray, kappa: float = cfg.KAPPA, **_) -> np.ndarray:
    """Fixed-fraction Kelly — no state dependence."""
    return np.full_like(d, kappa)


def half_kelly(d: np.ndarray, kappa: float = cfg.KAPPA, **_) -> np.ndarray:
    """Fixed half-Kelly fraction."""
    return np.full_like(d, kappa / 2.0)


def linear_ddr(
    d:     np.ndarray,
    kappa: float = cfg.KAPPA,
    b:     float = cfg.B,
) -> np.ndarray:
    """
    Linear (Grossman-Zhou) DDR:  f(d) = κ · d/b

    This is the proportional-to-cushion rule from GZ (1993),
    and the r → 0 normalised limit of the discounted closed form
    (Theorem 4.2, equation (9) of Sukhov 2026).
    """
    return kappa * np.clip(d, 0.0, b) / b


def exp_ddr(
    d:     np.ndarray,
    lam:   float = cfg.LAMBDA_STAR,
    kappa: float = cfg.KAPPA,
    b:     float = cfg.B,
) -> np.ndarray:
    """
    Exponential DDR (Definition 4.5):  f(d) = κ(1 − e^{−λd/b})

    Properties:
      * f(0) = 0  (full de-risk at barrier)
      * f(b) = κ(1 − e^{−λ}) < κ
      * strictly increasing, strictly concave in d
      * dominates linear DDR on penalised utility for calibrated λ*
    """
    return kappa * (1.0 - np.exp(-lam * np.clip(d, 0.0, b) / b))


def powerlaw_ddr(
    d:     np.ndarray,
    r:     float = 0.01,
    kappa: float = cfg.KAPPA,
    sigma: float = cfg.SIGMA,
    b:     float = cfg.B,
) -> np.ndarray:
    """
    Power-law DDR (Theorem 4.2):  f*(d) = κd / (1−α)

    This is the EXACT optimal control for the discounted HJB with
    discount rate r > 0. It is included for theoretical verification only.

    WARNING: this rule optimises a DIFFERENT objective (discounted, r=0.01)
    from the undiscounted (r=0) simulation. Do not compare its utility
    directly with the other strategies in Table 3.  See Remark 6.1.
    """
    alpha = alpha_discounted(r, kappa, sigma)
    return kappa * np.clip(d, 0.0, b) / (1.0 - alpha)


# ── Factory function ───────────────────────────────────────────────────────────

def make_policies(
    lambda_star: float = cfg.LAMBDA_STAR,
    kappa:       float = cfg.KAPPA,
    b:           float = cfg.B,
) -> dict:
    """
    Return the dict of policies used in the main comparison (Table 3).

    Only strategies that optimise the same undiscounted (r=0) objective
    are included in the primary comparison table.
    """
    return {
        "Full Kelly":          partial(full_kelly,  kappa=kappa),
        "Half-Kelly":          partial(half_kelly,  kappa=kappa),
        "Linear DDR (GZ)":     partial(linear_ddr,  kappa=kappa, b=b),
        f"Exp DDR (λ={lambda_star:.2f})":
                               partial(exp_ddr, lam=lambda_star, kappa=kappa, b=b),
    }


def make_lambda_sweep(
    lambda_grid: np.ndarray,
    kappa:       float = cfg.KAPPA,
    b:           float = cfg.B,
) -> dict:
    """Return one Exp DDR policy per λ in the grid (for lambda calibration)."""
    return {
        f"Exp DDR λ={lam:.3f}": partial(exp_ddr, lam=lam, kappa=kappa, b=b)
        for lam in lambda_grid
    }


if __name__ == "__main__":
    d = np.linspace(0, cfg.B, 9)
    print(f"{'d/b':>6}  {'LinDDR':>8}  {'ExpDDR':>8}  {'FullK':>8}")
    for d_val in d:
        arr = np.array([d_val])
        lin = float(linear_ddr(arr))
        exp = float(exp_ddr(arr))
        fk  = float(full_kelly(arr))
        print(f"  {d_val/cfg.B:4.2f}  {lin:8.4f}  {exp:8.4f}  {fk:8.4f}")

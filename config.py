"""
config.py — Central parameter store for Sukhov (2026).

All numerical experiments in the paper use the values defined here.
Changing a parameter here propagates automatically to all modules.
"""

import numpy as np

# ── Asset dynamics ─────────────────────────────────────────────────────────────
MU    = 0.08   # risky asset drift
R     = 0.00   # risk-free rate
SIGMA = 0.20   # volatility

# ── Derived quantities ─────────────────────────────────────────────────────────
KAPPA = (MU - R) / SIGMA**2          # Kelly fraction  κ = (μ−r)/σ²  = 2.0
DELTA = 0.20                         # drawdown tolerance
B     = -np.log(1.0 - DELTA)         # log-barrier width  b ≈ 0.2231

# ── Simulation ─────────────────────────────────────────────────────────────────
T_SIM      = 2.0         # horizon (years)
DT         = 1.0 / 252   # daily rebalancing
N_STEPS    = int(T_SIM / DT)
N_TOTAL    = 50_000      # total paths
N_TRAIN    = 35_000      # training split
N_TEST     = 15_000      # test split
SEED_TRAIN = 0
SEED_TEST  = 42

# ── Penalised utility ──────────────────────────────────────────────────────────
RUIN_PENALTY = -2.0      # p in U(π) = E[log W_T · 1{τ>T}] + p · P(τ≤T)

# ── Lambda calibration ─────────────────────────────────────────────────────────
LAMBDA_GRID_TRAIN = np.linspace(0.2, 3.0, 40)   # coarse sweep on train set
LAMBDA_STAR       = 0.888                        # calibrated optimum

# ── Sensitivity sweep ──────────────────────────────────────────────────────────
SENS_MU     = [0.04, 0.06, 0.08, 0.10, 0.12]
SENS_SIGMA  = [0.15, 0.20, 0.25, 0.30]
SENS_DELTA  = [0.10, 0.20]
SENS_T      = [1.0, 2.0, 3.0]
SENS_LAMBDA = np.linspace(0.2, 3.0, 10)
SENS_N      = 3_000      # paths per cell (speed vs accuracy trade-off)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
N_BOOTSTRAP = 500
CI_LEVEL    = 0.95

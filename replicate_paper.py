"""
replicate_paper.py — Master replication script for Sukhov (2026).

Runs all numerical experiments in the paper and saves tables and figures.
Expected runtime: ~10 minutes on a modern laptop (N=50,000 paths).

Usage
-----
    python replicate_paper.py              # full replication
    python replicate_paper.py --quick      # N=5,000 paths, fast check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial

import config as cfg
from survival import build_survival_table, alpha_discounted, leverage_discounted
from simulation import simulate, run_comparison, bootstrap_ci
from policies import make_policies, make_lambda_sweep, exp_ddr, linear_ddr
from sensitivity import run_sensitivity, fit_log_linear

OUT = Path("output")
OUT.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: styled print
# ─────────────────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — Survival-optimal leverage (Conjecture 3.2)
# ─────────────────────────────────────────────────────────────────────────────
def run_table1() -> None:
    section("Table 1 — Survival-Optimal Leverage")
    rows = build_survival_table()
    df   = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "Table1_survival_optimal_leverage.csv", index=False)
    print(f"\nSaved → {OUT}/Table1_survival_optimal_leverage.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Lambda calibration on training set
# ─────────────────────────────────────────────────────────────────────────────
def run_lambda_calibration(n_paths: int) -> float:
    section("Lambda Calibration (Train Set)")
    lambda_grid = cfg.LAMBDA_GRID_TRAIN
    us = []
    for lam in lambda_grid:
        pol = partial(exp_ddr, lam=lam)
        res = simulate(pol, n_paths=n_paths, seed=cfg.SEED_TRAIN)
        us.append(res["penalised_U"])
        print(f"  λ={lam:.3f}  U={res['penalised_U']:.4f}")

    us  = np.array(us)
    lam_star = float(lambda_grid[np.argmax(us)])
    print(f"\n  λ* = {lam_star:.3f}")

    # Save lambda curve plot
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(lambda_grid, us, "o-", ms=4, color="#2563eb")
    ax.axvline(lam_star, color="#dc2626", ls="--", lw=1.2, label=f"λ*={lam_star:.3f}")
    ax.axhline(max(us), color="grey", ls=":", lw=0.8)
    ax.set_xlabel("λ", fontsize=11)
    ax.set_ylabel("U(π)", fontsize=11)
    ax.set_title("Lambda calibration (train set, N=35,000)", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lambda_calibration.pdf", bbox_inches="tight")
    plt.close(fig)

    return lam_star


# ─────────────────────────────────────────────────────────────────────────────
# Table 3 — Test-set comparison
# ─────────────────────────────────────────────────────────────────────────────
def run_table3(lambda_star: float) -> None:
    section("Table 3 — Test-Set Comparison")
    policies = make_policies(lambda_star=lambda_star)
    rows = run_comparison(policies, n_paths=cfg.N_TEST, seed=cfg.SEED_TEST)
    df   = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "Table3_testset_comparison.csv", index=False)
    print(f"\nSaved → {OUT}/Table3_testset_comparison.csv")

    # Bar chart of penalised utility
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#dc2626", "#f97316", "#16a34a", "#2563eb"]
    names  = [r["Strategy"] for r in rows]
    utils  = [r["U(π)"]     for r in rows]
    ci_lo  = [r["CI_lo"]    for r in rows]
    ci_hi  = [r["CI_hi"]    for r in rows]
    yerr   = np.array([[u - lo, hi - u]
                       for u, lo, hi in zip(utils, ci_lo, ci_hi)]).T
    bars = ax.bar(names, utils, color=colors, alpha=0.85, yerr=yerr,
                  capsize=4, error_kw={"elinewidth": 1.2})
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Penalised utility U(π)", fontsize=10)
    ax.set_title("Test-set comparison (N=15,000, seed=42, 95% CI)", fontsize=9)
    plt.xticks(rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_testset_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure: leverage profiles
# ─────────────────────────────────────────────────────────────────────────────
def run_policy_figure(lambda_star: float) -> None:
    section("Figure — Leverage Profiles")
    d   = np.linspace(0, cfg.B, 300)
    r_disc = 0.01
    alpha  = alpha_discounted(r_disc, cfg.KAPPA, cfg.SIGMA)

    lin = linear_ddr(d)
    exp = partial(exp_ddr, lam=lambda_star)(d)
    plw = leverage_discounted(d, cfg.KAPPA, alpha, cfg.B)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d, np.full_like(d, cfg.KAPPA), "k:", lw=1.2,  label=f"Full Kelly κ={cfg.KAPPA}")
    ax.plot(d, lin, "-",  color="#16a34a", lw=2.0, label="Linear DDR (Grossman-Zhou)")
    ax.plot(d, exp, "-",  color="#2563eb", lw=2.0, label=f"Exp DDR λ*={lambda_star:.2f}")
    ax.plot(d, plw, "--", color="#9333ea", lw=1.5, label=f"Power-law (r=0.01, exact)")
    ax.set_xlabel("Log-distance to barrier  $d$", fontsize=11)
    ax.set_ylabel("Leverage  $f(d)$", fontsize=11)
    ax.set_title("Leverage rules as functions of $d_t$", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, cfg.B)
    ax.set_ylim(0, cfg.KAPPA * 1.05)
    ax.axvline(cfg.B, color="grey", ls=":", lw=0.8)
    ax.text(cfg.B * 0.98, cfg.KAPPA * 0.05, "$d=b$", ha="right", fontsize=8, color="grey")
    fig.tight_layout()
    fig.savefig(OUT / "fig_leverage_profiles.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT}/fig_leverage_profiles.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity sweep
# ─────────────────────────────────────────────────────────────────────────────
def run_sensitivity_analysis(n_paths: int) -> None:
    section("Sensitivity of λ* to Model Parameters")
    df  = run_sensitivity(n_paths=n_paths)
    df.to_csv(OUT / "Table_sensitivity_lambda.csv", index=False)
    print(f"\nSaved → {OUT}/Table_sensitivity_lambda.csv")

    fit = fit_log_linear(df)
    print(f"\nLog-linear fit: {fit['formula']}  (R² = {fit['R2']})")

    # Heatmap (T=2, two delta panels)
    mus    = sorted(df["mu"].unique())
    sigmas = sorted(df["sigma"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, delta_val, title in zip(axes, [0.10, 0.20],
                                    ["Panel A: δ=10%", "Panel B: δ=20% (baseline)"]):
        sub  = df[np.isclose(df["delta"], delta_val) & np.isclose(df["T"], 2.0)]
        grid = np.full((len(mus), len(sigmas)), np.nan)
        kgrd = np.full_like(grid, np.nan)
        for i, mu in enumerate(mus):
            for j, sig in enumerate(sigmas):
                row = sub[np.isclose(sub["mu"], mu) & np.isclose(sub["sigma"], sig)]
                if len(row):
                    grid[i, j] = row["lambda_star"].values[0]
                    kgrd[i, j] = row["kappa"].values[0]

        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r",
                       vmin=0.2, vmax=3.0, origin="lower")
        ax.set_xticks(range(len(sigmas)))
        ax.set_xticklabels([f"σ={v:.2f}" for v in sigmas], fontsize=9)
        ax.set_yticks(range(len(mus)))
        ax.set_yticklabels([f"μ={v:.2f}" for v in mus], fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Volatility σ"); ax.set_ylabel("Drift μ")
        for i in range(len(mus)):
            for j in range(len(sigmas)):
                if not np.isnan(grid[i, j]):
                    is_base = (np.isclose(mus[i], 0.08)
                               and np.isclose(sigmas[j], 0.20)
                               and np.isclose(delta_val, 0.20))
                    ax.text(j, i, f"λ*={grid[i,j]:.2f}\nκ={kgrd[i,j]:.1f}",
                            ha="center", va="center", fontsize=7.5,
                            fontweight="bold" if is_base else "normal",
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="white", ec="black", lw=1.5)
                            if is_base else None)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="λ*")

    fig.suptitle(
        "Sensitivity of λ* to model parameters (T=2yr, N=3,000 paths)\n"
        "Bold box: baseline (μ=0.08, σ=0.20, δ=20%)",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_sensitivity_heatmap.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {OUT}/fig_sensitivity_heatmap.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use N=5,000 paths for fast development check")
    args = parser.parse_args()

    n_train = 5_000  if args.quick else cfg.N_TRAIN
    n_test  = 2_000  if args.quick else cfg.N_TEST
    n_sens  = 500    if args.quick else cfg.SENS_N

    if args.quick:
        print("[QUICK MODE] Using reduced path counts for speed.")

    t0 = time.time()

    run_table1()
    lam_star = run_lambda_calibration(n_paths=n_train)
    run_table3(lambda_star=lam_star)
    run_policy_figure(lambda_star=lam_star)
    run_sensitivity_analysis(n_paths=n_sens)

    print(f"\n{'='*60}")
    print(f"All done in {time.time()-t0:.1f}s.  Output in ./{OUT}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

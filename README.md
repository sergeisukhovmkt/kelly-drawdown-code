# Dynamic De-Risking under Drawdown Constraints
### Replication Code — Sukhov (2026)

**Paper:** "Dynamic De-Risking under Drawdown Constraints: Structural Properties
and Heuristic Rules for Kelly Betting"  
**Author:** Sergei Sukhov, Market Microstructure Research Lab  
**Contact:** research@mmrls.com · mmrls.com

---

## Structure

```
kelly_drawdown/
├── README.md
├── requirements.txt
├── config.py                  # All parameters in one place
├── survival.py                # Module 1: flat-barrier survival probability
├── simulation.py              # Module 2: moving-barrier Monte Carlo engine
├── policies.py                # Module 3: leverage rules (Kelly, GZ, Exp DDR, Power-law)
├── sensitivity.py             # Module 4: lambda* sensitivity sweep
└── replicate_paper.py         # Master script — reproduces all tables and figures
```

## Reproducing the paper

```bash
pip install -r requirements.txt
python replicate_paper.py
```

This will produce:
- `Table1_survival_optimal_leverage.csv`   (Table 1 in paper)
- `Table3_testset_comparison.csv`          (Table 3 in paper)
- `Table_sensitivity_lambda.csv`           (Section 6.2a)
- `fig1_policy.pdf`
- `fig2_validation.pdf`
- `fig5_sensitivity.pdf`

## Key results

| Strategy          | U(π)   | Survival | Mean ann. log-ret. |
|-------------------|--------|----------|--------------------|
| Full Kelly        | −1.984 | 0.5%     | —                  |
| Half-Kelly        | −1.226 | 32.9%    | —                  |
| Linear DDR (GZ)   | +0.071 | 100.0%   | 3.69% (369 bps)    |
| **Exp DDR λ=0.89**| **+0.082** | **100.0%** | **4.25% (425 bps)** |

Parameters: μ=0.08, σ=0.20, δ=20%, T=2yr, κ=2.0, N=50,000 paths.

## Requirements

Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.7, Pandas ≥ 2.0

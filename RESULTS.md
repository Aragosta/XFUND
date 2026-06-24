# Deep Momentum (XGBoost) — Results

Replication/extension of Han (2022), *Bimodal Characteristic Returns and Predictability
Enhancement via Machine Learning*, using XGBoost in place of the paper's DNN.

**Universe:** survivorship-bias-free Tiingo broad US equities, 808 tickers passing the
coverage filter, monthly 2000-01 – 2026-06. **OOS test:** 2011-01 – 2026-04.
**Construction:** equal-weight L/S, top/bottom 5%, rolling 60-month window, annual
retrain, 5-seed ensemble, time-blocked validation. Returns net of 10bp transaction cost.

## Pipeline

1. **Features (32):** 5 nMOM + 5 MMOM (paper Eq. 6-8) + ACCEL/VOL/POS momentum-dynamics
   + 6 FFD (López de Prado, AFML Ch. 5: uniform-d FFD level + ΔFFD slopes, leakage-free —
   d frozen on the pre-OOS training window, causal FIR filter) + 10 SIZE dummies.
2. **Classifier:** XGBoost `multi:softprob` 10-class (return deciles), ensembled over seeds.
3. **Reclassification:** RET = Σ pₖμₖ (law of total expectation) — the paper's best/most-robust criterion.
4. **Turnover control:** Gârleanu–Pedersen partial adjustment wₜ=(1−δ)wₜ₋₁+δw*ₜ, δ=0.5 (quadratic-cost-optimal).
5. **Multi-horizon:** second ensemble on the t+2 label (decile of r₍ₜ₊₁₎), score = μ¹+μ².

## Headline comparison (OOS 2011–2026)

| Strategy | Ann. Return | Sharpe | Max DD | Ann. Vol | Ann. Turnover |
|---|---|---|---|---|---|
| Bench zMOM12 L/S (raw momentum) | 16.4% | 0.77 | −49.1% | 21.2% | 683% |
| DM-RET L/S | 23.3% | 1.41 | −15.3% | 16.5% | 1456% |
| **DM-RET+GP L/S** (Sharpe champion) | 28.6% | **2.25** | **−11.1%** | 12.7% | 786% |
| **DM-RET2+GP L/S** (multi-horizon) | **31.5%** | 2.19 | −16.5% | 14.4% | **703%** |
| S&P 500 B&H | 13.9% | 0.98 | −23.9% | 14.2% | 0% |

**Multi-horizon persistence diagnostic:** mean cross-sectional corr(μ¹, μ²) **φ = +0.652**.

## Key findings

- **Crash-preserving return cleaning matters.** Replacing a ±50% return clip with the
  paper's regime-adaptive cleaning (drop >+300%/<−95%, winsorize at the cross-sectional
  1/99 pct) restored the bench's true −49% momentum-crash drawdown — its earlier "great"
  Sharpe was the clip hiding tail risk.
- **FFD features added real signal:** +0.45–0.55 Sharpe across all reclassification
  criteria. The fix that mattered was a *uniform* d (cross-sectionally coherent) plus
  memory-preserving ΔFFD slopes rather than a rolling mean of the level.
- **GP partial adjustment is the standout:** halved turnover (1456→786%) *and* raised
  Sharpe 1.41→2.25 — partial adjustment low-pass-filters the book, cutting vol while a
  persistent signal keeps return.
- **Inverse-vol sizing failed** (1.41→1.27): the model's σ_i is dominated by near-common
  class-level variance, and in this microcap universe the edge concentrates in higher-vol
  names — risk-parity fights the alpha.
- **Multi-horizon (t+2):** φ=0.65 confirms the signal is persistent, so GP already
  captured most of the risk-adjustment. The t+2 model adds return (31.5%) and cuts
  turnover (703%) but not Sharpe — exactly what high φ predicts.

## Caveats (before any live consideration)

- 5-seed ensemble — needs a 20-seed confirm before trusting the Sharpe levels.
- The universe coverage filter (price>$1 in ≥70% of life) is computed over the full
  sample → residual survivorship bias inflates the *levels* (gaps are likely real).
- Turnover ~700–800% is still far above the paper's 166%; not yet live-tradeable.
- `size` is a dollar-volume proxy, not true market cap.

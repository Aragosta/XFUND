# Deep Momentum (XGBoost) — Results & Thesis

Replication/extension of Han (2022), *Bimodal Characteristic Returns and Predictability
Enhancement via Machine Learning*, using XGBoost in place of the paper's DNN.

**Universe:** survivorship-bias-free Tiingo broad US equities, monthly 2000-01 – 2026-06.
**OOS test:** 2011-01 – 2026-04 (184 signal months). **Construction:** equal-weight L/S,
top/bottom 5%, rolling 60-month window, annual retrain, 5-seed ensemble, time-blocked
validation. Returns net of 10bp transaction cost.

**Engine & leakage controls:**
- Single engine: `BACKTEST.backtest` (`freq=12`, `lag=0`). Signal at month-end *m* → earns return *m→m+1*. No look-ahead.
- **Point-in-time eligibility** (`compute_eligibility`): price>$5, ≥70% coverage over trailing 36 months, top 30% by dollar volume cross-sectionally, **and** absolute floor ≥$1M/month.
- **Delistings book −30%** (Han §4.1) via synthetic price at pos+1; `rets_monthly` derived from the same delist-adjusted prices so training labels and PnL are consistent.
- **OOS prediction cross-section = training cross-section** (`pool[t][0]`).

## Final pipeline

1. **Features (32):** 5 nMOM + 5 MMOM (paper Eq. 6-8) + ACCEL/VOL/POS momentum-dynamics
   + 6 FFD (uniform-d, causal FIR, d frozen on initial training window) + 10 SIZE dummies.
2. **Model:** XGBoost `multi:softprob` 10-class, ensembled over seeds.
3. **Reclassification:** RET = Σ pₖμₖ (law of total expectation).
4. **Turnover control:** Gârleanu–Pedersen partial adjustment wₜ=(1−δ)wₜ₋₁+δw\*ₜ, δ=0.5.

## Headline comparison (OOS 2011–2026)

*5-seed ensemble. Universe avg ~561 tradeable names/month (top 30% by dollar volume, ≥$1M/month floor, price>$5).*

| Strategy | Ann. Return | Sharpe | Max DD | Ann. Vol | Ann. Turnover |
|---|---|---|---|---|---|
| Bench zMOM12 L/S | 22.6% | 0.89 | −51.4% | 27.4% | 898% |
| **DM L/S** (RET reclassification) | **22.6%** | **1.11** | **−21.6%** | 20.3% | 1672% |
| DM-GP L/S (δ=0.5) | 18.6% | 1.07 | −21.5% | 17.4% | 894% |
| S&P 500 B&H | 13.9% | 0.99 | −23.9% | 14.2% | 3% |

**Key finding:** DM's edge over the raw momentum bench is **risk reduction, not return enhancement** — same 22.6% return but vol cut from 27% to 20% and max DD from −51% to −22%. The bimodality crash is dramatically compressed. GP at δ=0.5 over-smooths in this liquid universe: α-decay (−4pp return) exceeds the transaction-cost saving from halved turnover; deriving δ from the signal autocorrelation (φ=0.65) is the principled next step.

*(Previous stale headline — leaky pipeline, micro-cap universe, standalone backtester — showed Bench 0.77 / DM 1.62 / DM-GP 2.29. Tighter eligibility and pipeline fixes brought these to more credible levels.)*

---

# Thesis — what worked, what didn't

The investigation tested a long sequence of enhancements. A sharp pattern emerged:
**every win was on the robust *mean* signal or *downstream* of it; every attempt to
extract more from the predicted distribution's *shape* failed.**

## What worked

- **Crash-preserving return cleaning.** Replacing a ±50% return clip with the paper's
  regime-adaptive cleaning (drop >+300%/<−95%, winsorize at the cross-sectional 1/99 pct)
  restored the bench's true −49% momentum-crash drawdown. The clip had been silently
  amputating short-side crash risk — the bench's earlier "great" Sharpe was an artifact.
- **FFD features (+0.45–0.55 Sharpe).** López de Prado fractional differentiation added
  real signal — but only after two fixes: a **uniform d** (cross-sectionally coherent, so
  z-scores are comparable) and **memory-preserving ΔFFD slopes** instead of a rolling mean.
- **Gârleanu–Pedersen partial adjustment — the standout.** Quadratic-cost-optimal turnover
  control halved turnover (1456→786%) **and** raised Sharpe 1.41→2.25. Partial adjustment
  low-pass-filters the book: it cuts portfolio vol while a persistent signal keeps return.
  This was the single biggest, most robust, theoretically-grounded win.

## What didn't work

- **Inverse-vol (risk-parity) sizing** (1.41→1.27): the model's σ_i is dominated by
  near-common class-level variance, and in this microcap universe the edge concentrates in
  higher-vol names — risk-parity fights the alpha.
- **Multi-horizon (t+2) prediction:** the term-structure persistence φ=corr(μ¹,μ²)=+0.65
  showed the signal is persistent, so GP (which *assumes* geometric decay) had already
  captured the benefit. The explicit t+2 model added return but not Sharpe, and was a loose
  superposition of two persistence mechanisms — not a clean extension. Removed.
- **Distributional ranking — failed three ways (the central negative result):**
  - *Borda* (rank by P(rᵢ>rⱼ) over the distributions): **linear** in p → a cousin of the
    mean → collapsed to RET.
  - *Stochastic dominance (SSD)* on the decile softmax: nonlinear, but ranked on noise →
    Sharpe 0.78 (bench level), −58% drawdown.
  - *SSD on a calibrated quantile-regression head* (Axis 1, the proper fix): **still** failed
    (0.87 vs DM 1.21), and the quantile head itself underperformed the classifier
    (DM-GP 1.79 vs 2.25, −38% vs −11% DD).

## The unifying conclusion

> The **mean** (expected return, RET = Σ pₖμₖ) is the robust, rankable signal. The
> predicted distribution's **shape** (variance, tails, bimodality) is *not* a reliable
> cross-sectional signal — confirmed across both a decile-softmax and a calibrated
> quantile head, so the failure is the signal's intrinsic noisiness, **not** the
> representation. The classifier + law-of-total-expectation is also the *most robust* way
> to estimate that mean — it beat direct quantile regression, empirically vindicating the
> paper's choice of classification over regression (§3.1).

Han's bimodality is real **as a phenomenon**, but it cannot be exploited by re-weighting
or re-ranking the predicted probabilities. The wins live in (a) the robust mean signal and
(b) cost-aware portfolio construction (GP). The only remaining lever that could plausibly
beat RET is **architectural** — a relational/temporal model with a ranking loss and
turnover penalized in the objective (Axis 2/3), which improves the *mean ranking* and
lowers turnover *by design* rather than post-processing a distribution whose shape we have
now shown to be unreliable.

## Caveats (before any live consideration)

- 5-seed ensemble — needs a 20-seed confirm before trusting the Sharpe levels.
- ~~Universe coverage filter computed over full sample → survivorship bias.~~ **Fixed:**
  membership now uses point-in-time causal eligibility (trailing-window coverage at the
  signal date). Levels should drop somewhat once re-run.
- ~~Delisting/crash returns silently zeroed → optimistic short-side / crash-immunity.~~
  **Fixed:** delistings now book −30% in the price series, captured by the engine.
- ~~Headline produced by a standalone backtester, not the main engine.~~ **Fixed:** all
  PnL routes through `BACKTEST.backtest` (monthly, `lag=0`).
- Turnover ~790% (old number) is still well above the paper's 166%; not yet live-tradeable.
- `size` is a dollar-volume proxy, not true market cap.
- Delisting return is a flat −30% fallback (no CRSP delist returns in Tiingo monthly);
  a name truncated for benign data reasons is also charged −30%, a mild conservative bias.

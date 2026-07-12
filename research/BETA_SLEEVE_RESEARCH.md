# Vol-Managed Beta Sleeve — parked research direction (resume later)

**Status:** PARKED (2026-07-12). Removed from the deployed book by decision to keep the alpha pure/market-neutral.
The research is sound and reversible — this note is the pickup point.

## What it is
A directional sleeve that **holds the market index (SPY), sized inversely to volatility** — the *only* net-long
component (MOM/DM are market-neutral). Construction (leak-free):
- estimate market vol σ from recent daily returns (monthly grid)
- target vol = expanding median of realized vol
- **exposure = target_vol / recent_vol, capped 2.5×**  (low vol → hold more; high vol → hold less)
- sleeve return = exposure × next-month SPY return
Code: `research/beta_sleeve.py` (build + ERC combine), `research/beta_harvest.py` (the distribution/regime analysis).

## Why it works (evidence)
- Market Sharpe is concentrated in LOW-vol regimes (vol-quintile fwd SR: Q1 1.32 → Q5 0.58, same ~12% return).
  Vol-managing SPY: SR 0.77→0.91, maxDD -51%→-35% (1994+). Moreira-Muir volatility-managed portfolios.
- Direction is NOT timeable (market autocorr ~0; regime classifier + TSMOM both failed) → harvest STATICALLY,
  vol-managed, NOT by timing. See [[beta-sleeve-vol-managed]], [[tsmom-single-stock-fails]].

## Results (net, 2011+, ERC with MOM+DM)
- BETA standalone: SR 1.03, maxDD -18.4%, skew -0.74; participates in flat bull years (2013 +38%, 2017 +37%).
- corr to MOM+DM blend = **-0.03** (orthogonal diversifier).
- ERC MOM+DM = 1.55 → **ERC +BETA = 1.85**, maxDD -13.6%→-8.8%; fixes flat years, keeps 2022 crisis alpha (+24%).
- Dispersion-tilt (lean beta up in low-dispersion regimes): tilt=1 → 1.87, marginal SR, +return. (Also parked.)

## Why parked (the con)
Re-introduces market beta: portfolio corr to SPY rises -0.03 → **+0.28**; gives back in bears (2022 +52%→+24%);
1.03 is bull-flattered (no 2008-style bear in-sample; 1994+ = 0.91). Mandate choice: total return vs market-neutral
purity. We chose PURITY → removed the sleeve.

## How to resume
1. Regenerate: `DM_PKL=/tmp/mhdm_weights_s5_base.pkl PYTHONPATH=. python3 research/beta_sleeve.py`.
2. Beta sleeve = 3rd ERC stream alongside MOM+DM (the meta-model/stop live UNDER the alpha book; beta stays separate).
3. Open upgrade: size the exposure with the XGBoost **vol forecaster** (research/vol_overlay.py, OOS corr 0.89) instead of
   trailing vol. And re-test the dispersion-tilt out-of-sample before trusting its +return.
4. NOTE: vol-scaling the ALPHA sleeves HURTS (our alpha thrives in high vol); vol-management is ONLY for the beta leg.

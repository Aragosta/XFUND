# MOM sleeve — FUNDAMENTAL papers

Drop the PDFs in this folder. These are the load-bearing references for the momentum sleeve — the ones that
actually determine our design choices. Tiered by how fundamental they are. **Mandatory reads (per
[../../RESEARCH_PROTOCOL.md](../../RESEARCH_PROTOCOL.md) Phase R) are marked ★MANDATORY.**

---

## TIER 1 — THE ARCHITECTURE (why the sleeve is shaped the way it is)

1. **★MANDATORY — Han, C. (2022). "Bimodal Characteristic Returns and Predictability Enhancement via Machine
   Learning."** (Working paper; shared as `33053-2.pdf`.)
   *THE foundation.* Momentum forward returns are **bimodal**; the fix is **classify → RET = Σ pₖμₖ
   (reclassification) → rank**. The reclassification IS the crash/squeeze filter and it is a **return-space**
   property → **no liquidity filter needed** (full CRSP universe). Long-only beats L/S under realistic costs.
   Everything in `../../MOM_research.md`'s CURRENT CHAMPION descends from this paper.

2. **Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock
   Market Efficiency." Journal of Finance 48(1), 65–91.**
   The momentum anomaly itself — the premium we are harvesting exists and persists. Defines the 12-1 lookback and
   the winner/loser sort our features (`zMOM{1,3,6,9,12,18}`) are built on.

3. **Daniel, K. & Moskowitz, T. J. (2016). "Momentum Crashes." Journal of Financial Economics 122(2), 221–247.**
   WHY the short side is dangerous: losers are effectively **short a call** on the market (optionality only in
   bear+rebound) → **momentum crashes** in recoveries. This is the economic content of the bimodality — the reason
   `tval`-short blows up and why the short leg is "the tax." Motivates crash/vol management and the long-only tilt.

## TIER 1 — THE LOSS (why histogram + cross-entropy, not MSE)

4. **★MANDATORY — Farebrother, J. et al. (2024, DeepMind). "Stop Regressing: Training Value Functions via
   Classification for Scalable Deep Reinforcement Learning." ICML 2024 (arXiv:2403.03950).**
   On **fat-tailed, non-stationary** targets, **MSE's gradient explodes on the tails** while **cross-entropy over a
   histogram is bounded** → optimisation stability. The win is the LOSS, not the architecture. This is why we
   predict a histogram + CE and read RET back out.

5. **Imani, E. & White, M. (2018). "Improving Regression Performance with Distributional Losses." ICML 2018
   (arXiv:1806.04613).**
   The **HL-Gauss** origin — soft Gaussian-binned labels + cross-entropy. Companion to #4; the concrete recipe for
   our soft-two-hot / Gaussian-centred histogram target.

---

## TIER 2 — RISK MANAGEMENT & FEATURE PROVENANCE (design details we actually use)

6. **Barroso, P. & Santa-Clara, P. (2015). "Momentum Has Its Moments." Journal of Financial Economics 116(1),
   111–120.**
   **Volatility-managed momentum** — scaling exposure by realised vol removes most crashes and ~doubles the Sharpe.
   The basis for vol-targeting the sleeve at the STATE/risk layer rather than trying to time direction.

7. **Blitz, D., Huij, J. & Martens, M. (2011). "Residual Momentum." Journal of Empirical Finance 18(3), 506–521.**
   Momentum on **factor-residual** returns (market-beta-neutralised) — lower crash risk, higher risk-adjusted
   return. The provenance of our `resmom` feature and the residual-target experiments.

8. **Moskowitz, T. J., Ooi, Y. H. & Pedersen, L. H. (2012). "Time Series Momentum." Journal of Financial Economics
   104(2), 228–250.**
   **Time-series (trend) momentum** vs cross-sectional. Framing for TS-vs-CS; our verdict (see log) is that
   single-stock TS is weak and directional trend belongs at the index level — this paper is the reference point.

---

## TIER 3 — CONTEXT / BENCHMARK (read to situate, not to derive design)

9. **Gu, S., Kelly, B. & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." Review of Financial
   Studies 33(5), 2223–2273.**
   The ML-asset-pricing benchmark Han measures against (NN with ~900 covariates). Context for "features vs model":
   Han beats it with a handful of momentum features + reclassification — i.e. the MODEL/target is the lever, not
   feature count.

---

*Add a paper here only if it changes a design decision. Keep it fundamental — this is not a bibliography dump.*

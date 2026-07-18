# MR (mean-reversion) sleeve — FUNDAMENTAL papers

Drop PDFs here. The MR sleeve is framed as a **mixture-separation / classification** problem with a **speed-aware,
distributional target** (see `../../MR_research.md` "THE FRAMEWORK"): oversold events are a mixture of liquidity-driven
bounces vs information-driven "falling knives"; an ML classifier separates them, and the target should encode reversion
SPEED (vol→fast, turnover→slow). ★MANDATORY = read before proposing/testing (Phase R).

---

## TIER 1 — MECHANISM + SPEED (why reversion exists, bounce-vs-knife, and what sets its speed)

1. **★MANDATORY — Dai, Medhat, Novy-Marx & Rizova (2023). "Reversals and the Returns to Liquidity Provision."
   Financial Analysts Journal 80(2) (NBER w30917).** Reversal = liquidity-provision return; **high volatility → faster,
   larger reversals; low turnover → slower, longer-lived reversals.** The academic basis for a SPEED-weighted target and
   for vol/turnover as first-class features.

2. **★MANDATORY — Da, Liu & Schaumburg (2014). "A Closer Look at the Short-Term Return Reversal." RFS 27(1).**
   The **non-fundamental (liquidity) component reverts; the fundamental (information) component does not** = the
   bounce-vs-knife split the classifier is built to separate. The backbone of the ML thesis.

3. **★MANDATORY — Nagel, S. (2012). "Evaporating Liquidity." RFS 25(7).** Short-term reversal is compensation for
   **providing liquidity**; returns spike with VIX/illiquidity → the economic source AND why the STATE-stress gate matters.

## TIER 1 — the raw premium exists

4. **Jegadeesh (1990), "Evidence of Predictable Behavior of Security Returns," JF 45(3).**
   **Lehmann (1990), "Fads, Martingales, and Market Efficiency," QJE 105(1).**
   **Lo & MacKinlay (1990), "When Are Contrarian Profits Due to Stock Market Overreaction?" RFS 3(2).**

## TIER 2 — construction, tools, ML conditioning

5. **Cheng, Hameed, Subrahmanyam & Titman (2017). "Short-Term Reversals: The Effects of Past Returns and Institutional
   Exposure." Journal of Finance 72(2).**
6. **Ignashkina, Rinne & Suominen (2022). "Short-term reversals, returns to liquidity provision and the costs of
   immediacy." Journal of Banking & Finance 138.**
7. **Lo, A. (1991). "Long-Term Memory in Stock Market Prices." Econometrica 59(5)** (+ Hurst 1951) — the Hurst exponent
   as a rigorous mean-reversion-vs-trend (Liq-vs-Info) separator.
8. **Avellaneda & Lee (2010). "Statistical Arbitrage in the U.S. Equities Market." Quantitative Finance 10(7).**
9. **Gu, Kelly & Xiu (2020). "Empirical Asset Pricing via Machine Learning." RFS 33(5).** ML conditional > unconditional;
   reversal is a top feature. (Shared with MOM.)
10. **Farebrother et al. (2024) "Stop Regressing" + Imani & White (2018) HL-Gauss** — the distributional-target loss;
    applies here too (5-day reversion return is fat-tailed → histogram-CE over MSE). Copies in `../mom/`.

## TIER 3 — the source strategy being replicated/tested

11. **Quantitativo QP-ML series** (web, ★MANDATORY): "a mean-reversion strategy from first principles" (QP indicator +
    gate + event exit); "machine learning and the probability" (XGBoost P(bounce≤5d) = the mixture-separating filter);
    "long and short mean reversion machine" (L/S build; we replace its VIX filter with STATE stress).

---

*Add a paper here only if it changes a design decision. Keep it fundamental.*

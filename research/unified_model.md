# Unified Return Model — design note

Status: research / not yet built. Captures the architecture we converged on for merging
the cross-sectional (DM) and time-series (trend) signals into one pooled model.

## Thesis

Cross-sectional (XS) and time-series (TS) prediction share **one target** — next-period
return `r_{i,t+1}` — and differ only in (a) what you condition on and (b) how the portfolio
layer aggregates the forecast. So one model can serve both.

Decompose:

    r_{i,t+1} = β_i · f_{t+1}  +  ε_{i,t+1}
                (common / level)   (specific / relative)

- **Time-series momentum** (Moskowitz-Ooi-Pedersen 2012) trades the *level*: predict
  E[r_{i,t+1}], size directionally → carries market/factor exposure.
- **Cross-sectional momentum** (Han DM) trades the *relative*: predict
  E[r_{i,t+1}] − mean_j E[r_{j,t+1}] → dollar-neutral L/S.

Both are **linear functionals of the same conditional expectation**
`E[r_{i,t+1} | own history, cross-section]`. Estimate that once → read off both books.

This lines up with the momentum-decomposition paper (De Boer-Gao-Montminy 2025) we tested in
[pca_momentum.py](pca_momentum.py) / [rigorous_factor.py](rigorous_factor.py):
- common (industry/style) momentum ≈ persistent **level/trend** component (TS-flavored),
- stock-specific ≈ **cross-sectional** component that reverses short-term (DM's hedge).

We are not bolting two strategies together — we are modeling one return, two readouts.

## Where each existing sleeve sits

- **DM (current)** = cost-gated **short-term reversal hedge** on the stock-specific component.
  Lives in the *liquid* cross-section; degrades when microcaps are admitted (see below).
- **Industry/common momentum** (rigorous_factor) = persistent, **lower-turnover, higher-capacity**
  offense. The leg that actually scales. (Certification in progress.)

## Architecture sketch

- **Backbone:** decoder-only transformer (or TCN/LSTM) over each name's *vol-normalized*
  return sequence (Lim-Zohren-Roberts). Pooled across all stocks + markets (Han-international,
  Gu-Kelly-Xiu, StockGPT: pool, don't fragment). Market/country + entity embedding token for
  mild specialization.
- **Cross-sectional conditioning:** cross-attention over the panel at time t (or inject Han's
  MMOM cross-sectional means as macro-state tokens), so each forecast is aware of both the
  stock's own path (TS) and its rank vs peers (XS).
- **Head:** predict next-period return *distribution* (keeps the uncertainty idea; SRP/variance
  remain computable via Han Eq. 23-24).
- **Two readouts from one head:**
  - demean across panel → listwise ranking loss → DM/relative book,
  - level → Sharpe/directional loss → TSMOM/trend book,
  - turnover penalty inside the objective (the Gârleanu-Pedersen win, by design not post-hoc).
- **Portfolio layer** emits both a dollar-neutral sleeve and a directional sleeve.

## The one design change from today's DM

Current DM z-scores momentum cross-sectionally (Han Eq. 8) *specifically to strip the market
/ TS component*. To keep **both** signals we must NOT normalize the level away: feed raw
(vol-normalized) returns and make the cross-section a **conditioning input**, not a
normalization of the target. Consequence: the TS leg carries market beta → the unified book
is no longer market-neutral (AQR "everywhere" profile, Asness-Moskowitz-Pedersen). Explicit
and acceptable; the trend leg is the lower-turnover, higher-capacity complement to reversal.

## Open questions / experiments

1. **Decouple train vs trade universe** (the "more data" question). Train the pool on the broad
   universe, trade only the liquid book. Tests whether more *training* names help once you stop
   *trading* the microcaps. Small change to `_generate_all_dm_weights` (separate eligible_train
   from eligible_trade). Resolves the loose-filter regression:
     - loose-both ($1M, 3729 names): ann 19.2% / SR 1.00 / maxDD -39.8% / turn 1582%
     - tight-both (~561):            ann 22.6% / SR 1.11 / maxDD -21.6% / turn 1672%
   loosening BOTH hurt (return down, drawdown ~doubled) — microcap distribution shift + bounce.
2. **Certify structural momentum** (rigorous_factor): DONE — thesis INVERTED on 2011-2026 liquid US.
   - Common (FF5+industry) 12-1: ann −5.6% / SR −0.27; HAC **alpha −8.3%/yr (t=−2.80), MOM β=1.04** →
     it IS the momentum factor, with negative alpha. Industry/common momentum is NOT the offense here.
   - Stock-specific (residual) 12-1: ann +9.9% / SR **0.79** / maxDD −34%; HAC **alpha +8.8%/yr (t=+2.31),
     MOM β=0.43, R²=0.29** → genuine, largely-orthogonal alpha. **Residual momentum IS the offense**
     (Blitz-Huij-Martens 2011). Caveat: 5% (not 1%) significance, single regime, crude linear 12-1.
   → CORRECTION to the head spec: Head B target is the RESIDUAL (stock-specific) return at intermediate
     horizon, NOT the common component. Both heads live in the residual space, split by horizon —
     which vindicates the original t+1/t+2 multi-output instinct over the common-vs-specific split.
   → NEXT BUILD: "Deep Residual Momentum" = DM's full feature engine (multi-window mom, ACCEL/VOL/POS,
     size, and FFD on a reconstructed residual price path) run on residual returns, nonlinear XGBoost +
     RET. The rigorous_factor sleeve was crude linear 12-1; enriching it is the upside.
3. **Nonlinear spanning** (Level-2 ML alpha test): regress a sleeve on a boosted model of the
   base factors + lags/interactions; does the linear alpha survive? Cheap robustness check.
4. **Sequence vs tabular:** does a transformer on raw vol-normalized returns beat the XGBoost
   tabular DM on the *liquid* universe, net of realistic cost? (StockGPT's headline is
   gross/microcap; the honest test is net, liquid, at capacity.)

## Sequencing (earn the big model)

1. Certify the structural momentum offense (cheap; if no alpha, no architecture saves it).
2. Confirm DM as the cost-gated reversal hedge on the liquid universe.
3. Then build the unified pooled sequence model — one forecast of r_{i,t+1}, two readouts.

Building the big model before (1)-(2) is how you get a 6.5-Sharpe backtest that is really
microcap bid-ask bounce.

## References

- Han (2022) Bimodal Characteristic Returns (DM base).
- De Boer, Gao, Montminy (2025) momentum decomposition (common vs specific).
- Moskowitz, Ooi, Pedersen (2012) Time-Series Momentum.
- Asness, Moskowitz, Pedersen (2013) Value and Momentum Everywhere.
- Lim, Zohren, Roberts (2019) Deep Momentum Networks.
- Gu, Kelly, Xiu (2020/2021) ML asset pricing / autoencoder factors.
- StockGPT (2024) decoder-only transformer on raw returns.
- Nagel (2012) short-term reversal as liquidity provision.
- Hou, Xue, Zhang (2020) Replicating Anomalies (microcap concentration).

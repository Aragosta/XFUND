# MOM_research — working thesis & test log (MOMENTUM sleeve)

**Living document.** The current working thesis is at the top; every test is appended to the LOG below with its
EXACT method and its conclusion. Update this file whenever a MOM test is run — same discipline as the
[RESEARCH_PROTOCOL](RESEARCH_PROTOCOL.md) (READ → UNDERSTAND → DEFINE → MEASURE → ANALYZE). Sibling logs:
`VALUE_research.md`, `MR_research.md`.

---

## ⚠️ MANDATORY READING — read BEFORE proposing, testing, or claiming ANYTHING on this sleeve
Per [RESEARCH_PROTOCOL](RESEARCH_PROTOCOL.md) Phase R: every session, read this whole file **and** the two papers
below. Most "new ideas" and "bugs" this sleeve has hit were already answered (or pre-empted) by these two.
PDFs + full tiered reading list: [`papers/mom/`](papers/mom/README.md).

**1. Han (2022), "Bimodal Characteristic Returns and Predictability Enhancement via Machine Learning"** — the DM
paper; shared PDF `33053-2.pdf`. THE FOUNDATION of the DM/MOM architecture. Non-negotiable takeaways:
- Momentum forward returns are **bimodal** (winners stay/crash, losers stay/**squeeze**); the mean sits in the empty
  valley → argmax/point-MSE fail. The fix: **classify → RET = Σ pₖμₖ (reclassification, law of total expectation) →
  rank**. Reclassification-on-**Return** is the best of Han's 5 methods; ordinal ≈ nominal *after* reclassification.
- **The reclassification IS the crash/squeeze filter, and it is a RETURN-SPACE property.** A bimodal (small, volatile,
  squeezy) name has mass in BOTH *return* tails → moderate E[return] → pushed OUT of the extreme books (short-book avg
  cap $153M → **$1,589M** after Return-reclass). That is WHY Han uses **NO liquidity filter** — the universe is the
  **full CRSP cross-section (~3,837/mo)**, data-availability screens only. Excluding small firms is a *robustness check*
  only (SR 1.08–1.20; degrades but survives). **Do not add a dollar-volume pre-filter thinking it's "the champion
  universe" — it isn't; the reclassification is the filter.** (We burned a whole session on this before checking the paper.)
- **Corollary we derived:** the protection needs the histogram over **return**. MOM's **tval** histogram computes
  E[tval], NOT E[return] → it LOSES the short-side bimodality protection (tval L/S blows up; return-RET L/S survives).
  **tval helps the LONG book (−10pt DD); the SHORT leg wants return-RET.** They are different jobs.
- Under realistic (varying) costs Han's **VW L/S loses most profit (SR 0.22); the LONG-ONLY portfolio beats it (0.34)**
  — Han's own evidence that long-only is the cost-robust deployment (validates the long-only MOM sleeve). Size dummies
  are essential for value-weighting and drive the large-cap shift.

**2. Farebrother et al. (2024, DeepMind), "Stop Regressing: Training Value Functions via Classification for Scalable
Deep RL"** (+ Imani & White 2018, "Improving Regression Performance with Distributional Losses" = HL-Gauss). THE
FOUNDATION of the LOSS choice. Non-negotiable takeaway: on **fat-tailed, non-stationary** targets, **MSE's gradient
explodes on the tails** (∝ pred−y) while **cross-entropy over a histogram is bounded** (∝ |centre−pred|) → optimisation
stability. **The win is the LOSS, not the architecture.** This is WHY we predict a histogram + CE and read RET back
out, rather than MSE-regressing the return. HL-Gauss = the soft Gaussian-binned label version.

---

## CURRENT CHAMPION — `CURRENT BEST/mom_layer.py` (as of 2026-07-18)

**One object: a cross-sectional forecast of each name's forward TREND SIGNIFICANCE (tval), emitted as a monthly
per-name score the ERC/META layer ranks on.** DM's estimator (classify → RET reclassification) applied to MOM's
target (tval). Long-only by intent (discard-the-losers); L/S dollar-neutral available for comparison.

**CONFIG (exact):**
| component | value |
|---|---|
| **Target** | `tval@6` = slope ÷ SE of the forward 6-month log-price path (7 monthly points `[t..t+6]`, FULL path). **Single horizon** — no multi-horizon (T20/T21: blending doesn't raise SR). |
| **Representation** | histogram, **KB=10** bins (rank-bucketed) → `multi:softprob` → **RET = Σ pₖ·centerₖ** (DM reclassification). |
| **Centers** | **Gaussian quantiles** `norm.ppf((k+0.5)/10)` — free deterministic +0.03 SR vs uniform (T21). (emp-return centers tie → not needed, T23.) |
| **Model** | XGBoost `multi:softprob`, 10-class. `n_estimators=200, max_depth=5, lr=0.05, subsample=0.8, colsample=0.8`. **seeds=5** (prod ensemble). |
| **Features** | **full DM set** (`features.make_features`): Han `zMOM/MMOM`×{1,3,6,9,12,18} (12) + dynamics `ACCEL/VOL/POS` (6) + `SIZE_1..10` (10) + **TS trend** `hi52/trendR2/tsmom` (3) = **31**. FFD(+6) & MOM-pool(resmom/macd, +2) available via flags, DM-default OFF. Features z-scored (DM style); NOT re-Gaussian-ranked (rank transform ≈ cosmetic). |
| **Training** | rolling **72-month** window, refit **semiannually** (Jan/Jul), **embargo k−6** (labels fully realized before prediction — leak-free, audited T19). |
| **Universe** | liquid `eligibility(min_dollar_vol_pct=0.30, min_dollar_vol_abs=1e6)` — top-30% dollar-vol (same as DM). |
| **Construction** | cross-sectional (like DM). Native book = long top-decile + discard bottom; `backtest(ls=True)` = L/S dollar-neutral (long top-dec +1/n, short bot-dec −1/n). |
| **Engine** | `BACKTEST.backtest(pnl, freq=12, lag=0)`, tiered costs + borrow. Emits `out/mom_score.parquet` (months × names). |

**STATISTICS (all seeds=1 screens — the seeds=5 / full-universe validation is PENDING, killed mid-run 2026-07-18):**
| harness | book | rankIC | net SR | maxDD | turn |
|---|---|---|---|---|---|
| research (top-1000, synth/lag1) T21 | long-decile | ~0.054 | **~0.72** | −20 to −22% | 3.1 |
| DM engine (top-1000, pnl/lag0) T22 | long-decile | 0.312 | **0.44** | −27.8% | 3.5 |
| — same-footing DM (return target) T22 | long-decile | 0.381 | 0.39 | −37.4% | 4.4 |

**WHY these (evidence chain):** tval > return target on the BOOK, robust across harnesses (T19 SR .65>.52 full-univ;
T22 SR .44>.39 top-1000, **−10pt DD**), even though tval's IC-vs-raw-return is lower (it declines the delisting/lottery
tail = the source of its DD advantage; T23 — IC is NOT upliftable-by-design and shouldn't be, the BOOK is the judge).
Single 6-month horizon is the whole edge; representation/centers/multi-horizon/curvature are cosmetic-to-DD-only.

**OPEN GATE:** absolute numbers are harness-dependent (research synth/lag1 ~0.72 vs DM pnl/lag0 ~0.44); the
seeds=5 / full-universe production number is **not yet measured** (run killed). Confirm before "production-blessed."

---

## PRESERVED — the ASYMMETRIC 2-MODEL sleeve + SHORT-SQUEEZE theory (from `MOM.py`, deleted 2026-07-18)
`MOM.py` (the prior deployed champion) is retired in favour of `CURRENT BEST/mom_layer.py` (long-only tval sleeve).
Its **long side** is superseded, but its **asymmetric short head** is NOT replaced — so the full design is preserved
here so it can be rebuilt. Findings also in memory: [[two-model-short-target]] [[short-vol-thesis-proven]]
[[asymmetric-feature-map]] [[short-leg-is-the-tax]] [[mom-rankspace-champion]].

**THE THEORY — long and short are DIFFERENT prediction problems (asymmetric):**
- **LONG** = "which winners keep winning" — a **1st-moment return** problem. Symmetric return model works here.
- **SHORT** = "which losers keep falling **without squeezing**" — a **squeeze/2nd-moment** problem. The naive
  symmetric short (bottom decile of the return model) earns on average but **dies in squeeze outbreaks**:
  corr(short PnL, cross-sec squeeze rate) = **−0.76**; capping the short's upside flips its SR **0.28 → 1.69**.
  So the short leg is "the tax" — direction is right, but tail(squeeze)+borrow kill the symmetric version.
- **The fix = a squeeze-avoidance SHORT TARGET, not vol sizing.** Trend-scan t-stat (slope/SE of the forward
  log-price path) as the SHORT label: the **SE denominator is a squeeze filter** — it rewards steady decliners and
  penalises jagged, squeeze-prone names. Trend-scanning LOSES as a global/long target but WINS as the short target.

**THE IMPLEMENTATION (`MOM.py`, seeds=5, honest engine, full universe):**
- **Two multi-horizon Gaussian rank-space multi-output XGB regressors**, `HZ=(4,5,6)`, `REG = n_estimators=200,
  max_depth=6, lr=0.05, subsample=0.8, colsample=0.8`. Targets are `grank(...)` (Gaussian rank-transform).
  - **LONG model:** target = `grank(fwd return)[t+4,t+5,t+6]`; longs = top decile.
  - **SHORT model:** target = `grank(trend-scan t-stat)[t+4,t+5,t+6]`; shorts = bottom decile.
- **LONG/shared features (MFL):** ret/nret{3,6,12}, Baz-MACD composite (3 halflife pairs → y1/y2/y3 + `macd_comp`),
  FFD scores, amihud, dvtrend (log dv63/dv252), abnvol (21d/252d dollar-vol), resmom (11-1 residual, 60m β).
- **SHORT features (SFL) = MFL + the SQUEEZE AXIS:** `ivol` (63d realized vol), `upvol` (upside semis-dev),
  `MAX5` (mean of top-5 daily returns in 21d), `hi52` (price / 252d-max). These are the monotone, year-stable
  short predictors (feat_asym.py): ivol/upvol/MAX5/hi52/amihud/ret12.
- **Delisting:** −30% the month after a name's last listing. **Vol forecast** `volf` = EWMA(21d realized vol),
  shifted 1m (leak-free) — used as a SHORT FEATURE, not sizing.
- **DEAD-END kept behind a flag (`SHORT_VOLSCALE`, default OFF):** inverse-vol short sizing HURT — the
  lowest-forecast-vol shorts have POSITIVE forward returns (don't squeeze but don't fall either), so 1/vol
  weighting concentrates the short book on non-falling names (short leg +0.24 eq-wt → −0.18). Squeeze-avoidance is
  already done by the trend-scan target + ivol/upvol features.
- **Result (clean seeds=1, [[two-model-short-target]]):** short −0.11 → −0.03, combined 0.24 → 0.28 (modest). The
  2-model helps but the short leg stays a drag on the liquid universe → the honest baseline is **long-only /
  discard-the-losers** ([[remove-the-losers]]), which `mom_layer.py` now implements.

**OPEN for the short head (if revived):** trend-scan short target is the right idea but the net short SR is still
thin after borrow; the productive framing is EXCLUSION (discard bottom) + optional small squeeze-vetoed short, not a
full symmetric short book. Revisit only with a genuinely orthogonal short alpha or lower-cost borrow.

---

## THE WORKING THESIS (as of 2026-07-17)

**1. The alpha is loser-AVOIDANCE, and the outcome is BIMODAL.**
Momentum's edge is concentrated in avoiding the bottom 2–5% (bottom 2% ≈ −9.4%/yr), not in picking winners
(avoid-vs-select worth 2–10× more, growing with horizon). The reason it's hard: a name's forward return is
**bimodal** — a past winner either keeps winning or **crashes** (Daniel-Moskowitz: losers are effectively short
a call on the market, optionality only in bear+rebound); a past loser either stays down or **squeezes**. The mean
sits in the empty valley between the modes, so **point (MSE) regression predicts a value the stock rarely
realizes** — this is why MOM's regression short fails and DM's classification works.

**2. The modelling LEVER is the TARGET LOSS, not the architecture.**
Predict the **conditional distribution** of forward return as a **histogram over K buckets, trained with
cross-entropy** (HL-Gauss / two-hot / C51 / DeepMind "Stop Regressing"), and read a **continuous** score back out
(weighted bin centres). Why it beats MSE: the cross-entropy gradient is **bounded** (∝ |centre − pred|) while the
MSE gradient explodes on fat tails → **optimisation stability on noisy, non-stationary, heavy-tailed targets**
(the win is optimisation, not representation — Imani-White 2018; JMLR 2024). DM is the **coarse K=2 special
case** (P(top), P(bot)); the full histogram is strictly richer. Empirically the histogram is the **first model
all session to beat mean-regression** (see LOG T5/T6).

**3. Rank on RET; the SHAPE is NOT a tradable signal (CORRECTED per RESULTS.md).** DM's reclassification =
**RET = Σ pₖμₖ** (law of total expectation) — a robust estimate of the MEAN — and you RANK on RET. RESULTS.md's
central negative result: the distribution's SHAPE (variance, tails, bimodality) is NOT a usable cross-sectional
signal (Borda/SSD/quantile-head all failed) — only the mean ranks. So: **do NOT read lower-tail mass as a crash
signal** (that already failed). Deployment:
- LONG = top of the RET rank (the robust mean).
- LOSER-AVOIDANCE = **discard the BOTTOM of the RET rank** (not a separate tail read); never a symmetric short
  (short = tax; discard-bottom 0.61 SR / −30% DD beats market-neutral 0.48 / −52%). Manage beta separately.
- CRASH CONTROL = comes from the robust mean + **GP turnover control** (DD −51%→−22%, RESULTS.md) + STATE
  vol-target/coupling de-risk — NOT from tail-reading.
- SIZING = STATE vol-target dial (predicted-dispersion sizing lost — RESULTS.md inverse-vol failed).

**4. Target DENOISING — significance-weighting is the winning denoiser (SETTLED T15–T18).** The champion target
variable = **tval = slope/SE** (Schmerling center-weighted trend ÷ path-noise) on TOTAL returns — net SR **0.66**,
best across 4 harnesses. WHY: normalising the drift by its path-noise (the SE denominator) isolates the clean
trend. What does NOT help: RESIDUALISATION does not STACK with significance-weighting (substitutes, corr .93, even
hurts .66→.62 — T18); TREND-STRENGTH (unsigned R²) is a real premium but capacity-bound to low-IO/small → GONE in
our liquid universe (T17); vol-scaling/plain-slope/mixture-NLL all lose. (Residual momentum ~0.58 is a strong
CHEAP floor; the ML earns its keep only via the tval target — T16.)

**5. Learner & objective.** Trees now (they won). Cross-sectional **attention** (joint representation across
names) is the justified next architecture — coherent *because* the single distributional target removes the
reg-vs-cls task conflict that killed shared-trunk joint-rep, and the CE/histogram head is what lets a net scale
(Stop-Regressing). Objective = **rank-IC / Sharpe**, not point error (Poh LTR ~3×; Lim-Zohren direct-Sharpe).

**One-line framework (SETTLED, corrected):** *predict the histogram of forward tval → collapse to RET = Σ pₖμₖ
(law of total expectation, the robust mean) → RANK on RET → long top / discard bottom, beta managed separately,
GP turnover control, STATE vol-target sizing.* The "best of both" is NOT two operations — it is **DM's robust
estimator (classify→RET) applied to MOM's cleaner target (tval)**. One flow, one ranking (RET). The SHAPE/tails
are NOT a tradable signal (RESULTS.md); crash risk is compressed by the robust mean + GP + STATE, not tail-reading.

**Key literature:** Jegadeesh-Titman; Daniel-Moskowitz *Momentum Crashes*; Barroso-Santa-Clara; Blitz-Huij-Martens
*Residual Momentum* / Blitz-Hanauer *Idiosyncratic Momentum*; Gu-Kelly-Xiu; Imani-White *HL-Gauss* + JMLR-2024
histogram-loss study; DeepMind *Stop Regressing* (ICML 2024); C51/two-hot; Poh-Lim-Zohren *Learning to Rank*;
Lim-Zohren *Deep Momentum Networks*; Wood-Roberts-Zohren *Spatio-Temporal Momentum*; MQ-Forecaster (multi-horizon).

---

## TEST LOG (append-only; newest at bottom)

Format per entry: **ID · date · hypothesis · method (script, universe, seeds, metric) · result · conclusion.**

### T1 — 2026-07-17 · quantile-grid distribution vs mean-regression
- **Hypothesis:** predicting a quantile grid (a distribution) beats point mean-regression for ranking/loser-ID.
- **Method:** `research/unified_dist.py`, top-1000, seeds=1, OOS 2011+, cumulative-6m target. base = XGB
  multi-out mean on grank(fwd); Aq = XGB `reg:quantileerror` at τ={.1,.25,.5,.75,.9}, score=median, loser=−q.1.
  Metrics: rankIC, loser@bot5 (mean fwd-cumret of the 5% most loser-flagged; lower=better), top-decile long net SR.
- **Result:** base rankIC .0336 / longSR .54 ; Aq .0289 / .51.
- **Conclusion:** **REJECT.** Quantile grid (coarse, 5 knots) does not beat the point estimate.

### T2 — 2026-07-17 · torch 2-Gaussian mixture (true bimodal regression)
- **Hypothesis:** an explicit 2-component mixture-density net captures the bimodality and beats mean-reg.
- **Method:** same harness; Am = per-row MDN (2 Gaussians), NLL loss, 60 epochs. VALIDITY CAVEAT: never verified
  the two components separated (may have collapsed to unimodal).
- **Result:** Am rankIC .0195 / longSR .50 — worst rankIC.
- **Conclusion:** **REJECT.** Mixture-NLL is unstable (no gradient bound); the bimodal *model* is the wrong tool —
  the bimodal *loss* (histogram/CE) is (see T5). Do not pursue mixture-density regression.

### T3 — 2026-07-17 · DM's 2-bucket representation on our pool
- **Hypothesis:** DM's P(top)−P(bot) classification beats mean-reg (the "why DM works" claim).
- **Method:** same harness; dm2 = two XGB binary classifiers (top-decile, bot-decile), score=P(top)−P(bot).
- **Result:** dm2 rankIC .0300 / longSR .51 — ties/loses base on this compact pool.
- **Conclusion:** **NEEDS-MORE.** 2-bucket alone isn't the win here; the FULL histogram is (T5). Suggested DM is a
  coarse special case, not the ceiling.

### T4 — 2026-07-17 · cross-sectional attention (joint representation), torch
- **Hypothesis:** attention across names (peer-relative) + mixture head beats per-row.
- **Method:** `unified_dist.py` Bx = MLP encoder + self-attention over names/date + MDN head. RUSHED — killed
  mid-run; single seed, tiny net, grank inputs already inject cross-sectional info → muddied contrast.
- **Result:** inconclusive (not trusted).
- **Conclusion:** **NEEDS-MORE.** Re-do only after the representation (histogram) is confirmed; use a CE/histogram
  head (not mixture) so it can scale.

### T5 — 2026-07-17 · K-bucket HISTOGRAM (cross-entropy) vs mean-reg — ★ the win
- **Hypothesis:** predict the full conditional histogram (K=10) with cross-entropy; read continuous mean back out.
- **Method:** `unified_dist.py` hist = XGB `multi:softprob` num_class=10 on cross-sectional deciles of cumulative
  return; score = Σ P_k·centre_k; loser = P(bottom 2 buckets). Same top-1000/seeds=1/net-engine harness.
- **Result:** hist rankIC **.0396 / longSR .58** vs base .0336/.54 — **first arm all session to beat base.**
- **Conclusion:** **ADOPT (screen).** The distributional-histogram *loss* is the lever. Confirms the thesis.
  Needs seeds=5 / full-universe / real-engine confirmation.

### T6 — 2026-07-17 · multi-horizon SIMULTANEOUS histogram (one vector-leaf)
- **Hypothesis:** predicting the histogram at every horizon jointly (MOM-sense multi-output) beats single-horizon.
- **Method:** `research/hist_mh.py`, top-1000, seeds=1, H=(1,3,6). histMH = ONE `multi_output_tree`
  (binary:logistic) on soft TWO-HOT (H×K) labels, per-h normalise, rank = mean_h E[bucket]; also squeeze-aware
  read-off loserSQZ = P(bot2)−P(top2). vs hist (single cumulative-6m) and base (MH mean-reg).
- **Result:** base .0276/.51 ; hist .0389/**.60** ; histMH .0376/.55. loser@P .0554 < loser@SQZ .0591 (SQZ worse).
- **Conclusion:** **NEEDS-MORE / partially REJECT.** (a) Not a clean isolation — histMH changed horizon-set AND
  cumulative-vs-vector target at once; single-cumulative already encodes the horizon path. (b) squeeze-aware
  read-off HURTS long-exclusion (helps only for shorting). CLEAN isolation pending (T7).

### T7 — 2026-07-17 · histogram vs regular DM on DM's REAL harness  [IN PROGRESS]
- **Hypothesis:** swapping ONLY the target representation (DM's 2-bucket → K-bucket histogram) inside DM's exact
  full-universe harness (Han+TS features, 120-mo walk, top-q L/S, net engine) beats regular DM.
- **Method:** `DM.py` with `REP={dm|hist}` env (added; default dm = production untouched). labels_hist = soft
  two-hot over K=10 buckets per horizon; XGBRegressor multi_output_tree; score = mean_h E[bucket]. Seeds=1 screen
  first. THE BASELINE IS REGULAR DM. Also to run: clean multi-horizon isolation REP=hist H=(1,) vs H=(1,2,3).
- **Result:** REP=dm (regular DM): IC .0359 / net SR −0.32 / maxDD −82.7% / turn 10.2. REP=hist: IC **.0378** /
  net SR −0.36 / maxDD −84.8% / turn **8.5**. (seeds=1, full universe, symmetric top-5% L/S, net engine.)
- **Conclusion:** **NEEDS-MORE — representation win is MASKED by the short leg.** Histogram gives a small IC bump
  (+.002) and lower turnover but NO net-SR gain, because on the symmetric-L/S book the net SR / −83% DD are
  dominated by the SHORT leg (the squeeze tax), not by ranking quality — so a better ranking can't show through.
  This directly reinforces thesis pt 3: the histogram improves the RANKING (IC↑, and longSR↑ on the T5/T6 screen),
  but its value must be captured via LONG + exclusion deployment, NOT symmetric L/S. Seeds=1 is also noisy
  (Δ net SR −0.04 is within noise). NEXT: evaluate hist vs dm as a LONG-ONLY / exclusion book (T8), + seeds=5.

### T8 — 2026-07-17 · CLEAN multi-horizon isolation (histogram rep, only horizon count varies) — ★ MH confirmed
- **Hypothesis:** within the histogram representation, multi-horizon H=(1,2,3) beats single H=(1,) — clean
  isolation (same target family, universe, features, engine; ONLY the horizon count changes). Fixes T6's confound.
- **Method:** `DM.py` REP=hist ARM=both SEEDS=1, full universe. SH-DM=(1,), MH-DM=(1,2,3). Metric: IC (ranking).
- **Result:** SH-DM(1) IC **.0203** ; MH-DM(1,2,3) IC **.0378** (net SR both ~−0.5, short-leg-dominated — read IC).
- **Conclusion:** **ADOPT — multi-horizon NEARLY DOUBLES ranking IC (.020→.038).** The user's instinct was right;
  T6's "MH doesn't add" was a CONFOUNDED test (changed horizon-set AND cumulative-vs-vector at once). Simultaneous
  multi-horizon (one vector-leaf emitting all h) is a real, large lever. Multi-horizon is GOLD (confirms memory
  [[multihorizon-dm-and-ltr]] +0.18). NOTE net SR still masked by the symmetric short leg → T9 deployment.

### External research report integrated (2026-07-17) — refinements to the thesis
A commissioned lit review ("A Unified Momentum Sleeve", ~30 papers) CONFIRMS the thesis (distributional CE head;
target/loss > architecture; asymmetric lower-tail veto not symmetric short; trend-scan out) and adds testable
refinements: (1) **HL-Gauss GAUSSIAN-smoothed soft labels > HARD two-hot** — DeepMind Stop-Regressing shows hard
two-hot underperforms because it doesn't spread mass to neighbour bins; our T5–T8 used hard two-hot/softprob →
UPGRADE to Gaussian smoothing (~50–100 bins, σ≈1–2 bin widths). (2) **RESIDUAL + VOL-SCALED target** = highest-
leverage target change (Blitz: half vol same return; Barroso vol-scaling) — DM.py already has RESID/VOLADJ flags →
test now (T9). (3) **Triple-barrier / max-drawdown-within-horizon** as the AUXILIARY crash-veto target (not the
primary label; replaces trend-scan). (4) **Rank-IC / listwise auxiliary loss** for the long book (Poh LTR — treat
the ~3× claim with skepticism, our LambdaMART FAILED; LambdaRankIC/rank-IC-aligned only). (5) **DeePM causal-sieve
= STRICTLY-LAGGED cross-sectional attention** to avoid look-ahead across names — the right form for Stage-3.
(6) **Factor-momentum features** (Ehsani-Linnainmaa) high-value in US, provisional intl.
Where our data TEMPERS it: T7 shows the histogram's ranking gain does NOT convert to net SR on the SYMMETRIC L/S
book (short leg dominates) — so the doc's "adopt the head" is right for RANKING but net value REQUIRES the
long+exclusion deployment. Our session also positively CONFIRMS "target/loss > architecture" (histogram/CE won;
attention & mixture lost).

**STAGED PLAN (from the report, adopted):** Stage 1 = fix TARGETS (residual + vol-scaled + HL-Gauss soft-bin,
multi-horizon vector) — cheapest, highest leverage, do first. Stage 2 = lock FRAMING (CE head as shared rep +
crash veto + dispersion sizing; rank-IC auxiliary for long book; asymmetric lower-tail EXCLUSION deployment).
Stage 3 = ARCHITECTURE (causal-sieve cross-sectional attention with the histogram head; XGBoost as benchmark).

### T25 — 2026-07-18 · Borrowable-shorts filter (DATAHUB global) + vector L/S re-test — ★ short leg dies from the CRASH not borrowability; target winner is UNIVERSE-DEPENDENT
- **Q:** with the global borrow filter enforced (short only `hub.elig('shortable')`, mdv>$25M), does ANY L/S survive,
  and does the vector [return-hist, tval-hist] (asym long-tval/short-return, or sym blend) beat pure tval?
- **Method:** `mom_research.py` (harness) `multi_hist(['fwd6','tval'])`; DATAHUB tier=liquid (3,310 names), seeds=1,
  decile, BORROWABLE shorts only. Centralised the filter: mom_layer now uses `hub.elig` (removed its ad-hoc
  UNIVERSE.eligibility+min_dvol "loose filter"); short leg restricted to `hub.elig('shortable')` in backtest + net().
- **Result (IC6m / net SR / maxDD):** tval_LS .104/−.68/−97.9 ; ret_LS .086/−.67/−96.7 ; asym_LS .104/−.71/−97.8 ;
  sym_LS .097/−.64/−96.8 ; tval_long .104/**.16**/−36.5 ; ret_long .086/**.29**/−40.7.
- **Conclusion:** (1) **Borrow filter barely helped (−99.6→−97%); the short-leg killer is the MOMENTUM CRASH**
  (Daniel-Moskowitz: losers squeeze in rebounds), not borrowability — hits liquid large-cap shorts too, paid in full by
  the honest engine. **No filter/target/vector fixes the symmetric short leg** (tval/ret/asym/sym all −.64..−.71).
  **Vector-target / asymmetric-short idea = DEAD for L/S.** Long-only is the only viable book (shown T22/T24/T25).
  (2) ★ **TARGET WINNER IS UNIVERSE-DEPENDENT:** tval wins long-book only on the TIGHT top-1000 (T22 .44>.39); on the
  broader liquid-3310 (here .16<.29 return) and full penny (T24 .11<.26 return) **RETURN wins**. **tval is a
  MOST-liquid refinement, NOT a universal champion** — pin the DEPLOYABLE universe and test the target THERE before
  claiming a champion. Note tval keeps HIGHER IC (.104>.086) but LOWER book SR on the broad universe (+IC≠+book again).
  GATE: seeds=1. Filter now centralised in DATAHUB (global, all sleeves); DM.py still on UNIVERSE.eligibility (future consolidation).

### T24 — 2026-07-18 · Han-universe reconciliation + return/tval VECTOR target — ★ short leg ALWAYS needs borrowability; reclassification ≠ sufficient
- **Q:** Han (2022) uses NO liquidity filter (full CRSP) because the RET=Σpₖμₖ reclassification is a RETURN-SPACE
  squeeze filter (bimodal names → moderate E[return] → out of extremes). Can a VECTOR target [return-hist, tval-hist]
  keep tval's long edge AND recover Han's short-side crash immunity? (long=E[tval], short=E[return]; + symmetric blend).
- **Method:** `research/mom_research.py` (harness) `multi_hist(['fwd6','tval'])` = one multi_output_tree → RET_ret,
  RET_tval. Books: tval_LS / ret_LS / asym_LS (long tval, short ret) / sym_LS (blend) + long-only. FULL universe
  (min_dvol=0, Han's no-filter), seeds=1, decile, honest engine (borrow+squeeze-in-full). [was `vector_target.py`, now deleted]
- **Result (net SR / maxDD):** tval_LS −0.98/−99.6 ; ret_LS −1.02/−99.5 ; asym_LS −1.05/−99.7 ; sym_LS −0.97/−99.5 ;
  tval_long +0.11/−35 ; sym_long +0.19/−37 ; **ret_long +0.26/−37**. (IC nan: fwd6 non-finite on full penny universe.)
- **Conclusion — PREDICTION FALSIFIED.** ret_LS did NOT survive; ALL L/S blew up ~−99%, return-RET included. WHY:
  (1) we shorted UN-BORROWABLE penny stocks — Han's short side clears a HIGHER dollar-vol floor (borrowability); "no
  liquidity filter" is for the LONG universe, NOT the short book. (2) equal-weight + honest squeeze/borrow on tiny
  stocks is unsurvivable regardless of target (Han value-weights / reclass shifts to large-cap $1,589M). **So the
  reclassification is a real squeeze-avoidance mechanism but is NECESSARY-NOT-SUFFICIENT — the short leg ALWAYS needs a
  borrowability/liquidity constraint.** The vector asym/sym benefit is UNTESTABLE here (penny-short blowup masks it).
  **Survivors:** long-only is robust (all +); on the FULL universe **return-target wins long-only (.26>sym .19>tval .11)
  — OPPOSITE of the liquid universe (T22: tval wins)** → target choice is universe-dependent. **Next:** test vector
  asym/sym in a SURVIVABLE L/S setting (add borrowable-shorts filter + liquid/VW) before judging the vector idea.
  Reaffirms [[short-leg-is-the-tax]] and the long-only MOM design. GATE: seeds=1, IC metric broken on full univ.

### T23 — 2026-07-18 · Can we UPLIFT IC by design + keep the histogram? — ★ NO (both loss & readout fail); IC is the wrong judge (delisting-inflated)
- **Q:** DM's return target scores higher IC than tval on the DM engine (delisting-laden fwd6). Can we recover that IC
  from a tval histogram — by re-aiming the LOSS at decile thresholds, or the READOUT at empirical returns — without
  losing tval's book? (Same shared pool as T22: full-DM feats, top-1000, DM engine, seeds=1, 2011+.)
- **RECONCILIATION first** (`mom_vs_dm.py`, same 10-bin estimator, vary target): tval-h **.3124 IC / .44 SR / −27.8%** ;
  ret-hist **.3717 / .32 / −38.2%** ; DM-2bin .3809 / .39 / −37.4%. → the return target's higher IC does NOT need the
  2-bin estimator — it's the METRIC: raw fwd6 rank is dominated by the predictable −30% delisting/lottery tail. IC
  ordering FLIPS vs T19 (clean-return harness, tval higher IC) but the BOOK ordering is ROBUST both harnesses (tval SR/DD
  wins). **IC-vs-raw-return favors the return target here but buys a WORSE book.**
- **LOSS re-aiming** (`ordinal_hist.py`): tval_mc .3124/.44 ; **tval_ord (CORAL K-1 threshold heads) .2300/.45 (IC DROPPED)** ;
  tval_tw (tail-weighted CE) .3012/.45 ; tval_ord_tw .2442/.42. → **no uplift; CORAL hurts.** A loss can only sharpen what
  the TARGET contains — it can't manufacture IC about a quantity (raw-return tail) tval doesn't encode.
- **READOUT centers** (`centers_test.py`, RET=Σpₖμₖ, vary μₖ): int .3054/.45 ; gauss .3124/.44 ; emp_tval .3205/.42 ;
  **emp_ret (TRUE DM reclassification = causal per-decile mean RETURN) .3123/.44** → all tie within noise. Any MONOTONE
  centers give a near-identical cross-sec RANK, so the reclassification is COSMETIC for a rank-based long-decile book.
- **Conclusion:** **IC is NOT upliftable-by-design for a tval model** — loss (target-content gap, unfixable) and readout
  (monotone reweight, rank-invariant) both fail. And you WOULDN'T want to: the return target's IC edge = chasing the
  delisting tail = −10pt DD. **Low IC is a FEATURE (tval declines the lottery tail); the BOOK is the judge, tval wins it.**
  Sleeve unchanged: tval → histogram → RET → Gaussian centers (emp_ret theoretically cleaner but empirically identical =
  keep the simpler centers). Confirms [[quantile-edge-decomposition]] "+IC ≠ +decile". GATE: seeds=1.

### T22 — 2026-07-18 · MOM(tval) vs DM(return-decile) on IDENTICAL full-DM features + DM engine — ★ tval wins the BOOK (−10pt DD)
- **Q:** hold everything fixed (the 31-feature FULL DM set = make_features + TS; top-1000; XGB REGC; embargo k−6; DM
  engine pnl/lag=0), vary ONLY the target: does MOM's tval beat DM's return-decile on the deployable book?
- **Method:** `research/mom_vs_dm.py` builds ONE shared pool via `CURRENT BEST/mom_layer.MomLayer.build_pool()`, runs
  both estimators. MOM = tval@6 → hist → RET=Σpₖ·gaussian_centerₖ. DM = 6m-fwd-return → P(top-dec)−P(bot-dec).
  seeds=1, long-decile net, 2011+.
- **Result (IC6m / net SR / maxDD / turn):** MOM tval **.3124 / .44 / −27.8% / 3.5** ; DM return .3809 / .39 / −37.4% / 4.4.
- **Conclusion:** **tval wins the BOOK on every tradable metric** — +0.05 SR, **−10pt drawdown**, −0.9 turnover — while
  DM wins ONLY raw IC. That IC is **delisting-tail-inflated** (the return target keys on predictable −30% delisters →
  high rank-IC that does NOT convert to long-book value; classic "+IC ≠ +decile", [[quantile-edge-decomposition]]).
  Validates `mom_layer.py`'s tval choice on the SAME features/engine as DM. NOTE absolute levels here (SR ~0.4, IC ~0.3)
  are the DM-engine footing (pnl/lag=0/full-elig/delisting), NOT comparable to the research harness (synth/lag=1/top-
  1000-by-dvol, tval ~0.72) — only the MOM-vs-DM DELTA is valid. GATE: seeds=1; confirm at seeds=5.

### T21 — 2026-07-18 · Target anatomy sweep (5 studies): deflation, residual, honest multi-horizon, path-shape, representation — ★ alpha = linear-trend significance; everything else shapes DRAWDOWN
- **Q:** having fixed tval as target, probe every remaining lever: (a) does full noise-deflation over-penalize
  magnitude? (b) residualize the target? (c) does HONEST multi-horizon help (fix T20's collinear build)? (d) does
  forward-path curvature carry info beyond slope? (e) hist vs Gaussian-rank vs raw target representation, and their
  combination? All top-1000, rolling-72, seeds=1, RET long-decile net, 2011+, tval@[0,6] unless noted.
- **Scripts:** `target_deflate.py` `fwd_resid_tval.py` `mh_fix.py` `orthopoly_target.py` `target_transform.py` `transform_combo.py`.
- **(a) DEFLATION** `slope/SE^α`: α0(pure mag) .46 ; α0.5 .64 ; α1(tval) .69 ; α1.5 .71 ; cumret6 .47 ; cum*|tval|rank .62.
  **MONOTONE in deflation — magnitude IS the noise.** Multiplying tval by any return metric moves toward .46. REJECT magnitude composites.
- **(b) RESIDUAL forward-tval** (robust 60m PIT β, in the DEPLOYED pipeline, not T18's past-signal): total .70/−22.4% ;
  resid .69/−21.6% ; corr **0.899** (substitutes, matches T18). Whisker of DD, no SR. **REJECT** — significance-weighting already denoised.
- **(c) HONEST multi-horizon** (T20 was collinear: nested-tval adj corr .78–.94): dtval_disjoint (non-overlap [0,2]/[2,4]/[4,6])
  **.0554/.72/−20.2%** BEATS single TVAL(h6) .0540/.71/−23.0% ; skip_2b .66 ; disjoint-RETURN seg_0_5 .49 / seg_1_5 .43 (worst).
  **MH helps ONLY when pieces are BOTH independent AND denoised** — tval supplies denoising, disjoint windows supply independence;
  raw-return segments are independent but undenoised = poison. Gain is DD, not SR. (Corrects T20's "MH dead" — that was a build artifact.)
- **(d) PATH SHAPE** (orthonormal Legendre significance, t1⊥t2 = +0.002): slope-only(=tval) .0532/.72/−21.3 ;
  curv-only .0099/**.40** (near-noise) ; slope+curv **.0583/.71/−19.9%** ; +cubic .0534/.69/−21.3 (dilutes). **Saturates at 2 comps:**
  slope=alpha, curvature=DD refinement (−1.4pt), cubic=noise. Same signature as disjoint — 2nd orthogonal dim shapes DD, not alpha.
- **(e) REPRESENTATION** (hist vs gauss-rank vs raw, tval target): hist_ret .69 ; gauss_reg .66 ; raw_reg **.68 (did NOT blow up)**.
  For the tval target — already a tamed t-stat — representation is ~cosmetic (the "transform matters" thesis was for FAT-TAILED return
  targets). **COMBINATION** (transform_combo, same model two readouts): hist_uniform .69 vs **hist_gauss .72** (Gaussian-quantile RET
  centers) = **DETERMINISTIC free +.03 SR** (same model, tail-stretched centers reweight top-decile); hlgauss_soft .69 ; gauss_reg .66.
- **Conclusion:** **THE ALPHA IS THE LINEAR TREND SIGNIFICANCE (tval); every other lever shapes DRAWDOWN, not return.** Three
  independent routes all top out at ~.72 SR (hist_gauss, dtval_disjoint, slope-only) and all help DD. Bankable free win = **Gaussian-quantile
  RET centers** (deterministic). Optional DD refinements (not adopted yet): disjoint-tval MH and/or +curvature. REJECTED for good: magnitude
  composites, residual target, nested MH, disjoint-return MH, cubic+ path terms, point-regression rep. **GATE: seeds=1/top-1000 — the .69–.72
  band is within seed noise EXCEPT the hist_gauss readout (deterministic); confirm at seeds=5 before adopting the Gaussian centers.**

### T20 — 2026-07-17 · Full analysis: DM vs TVAL vs SIMULTANEOUS short-horizon vectors {2,2·3,2·3·4,2·3·4·5} — ★ single 6m is the whole edge
- **Q:** (1) does a SIMULTANEOUS multi-horizon short-window tval vector beat single TVAL(h6)? (2) where does it
  plateau 2→3→4→5? (3) DM vs TVAL(h6) on IDENTICAL footing?
- **Method:** `research/mh_full.py` top-1000 (~905 avg), rolling-72 window on ALL arms (fixed a prior
  inconsistency: mhvec was −72 while DM/single were expanding → now consistent + ~3× faster), RET=Σpₖμₖ ranked
  long-decile net, 2011+, seeds=1, embargo k−6. tval@h = slope/SE of fwd log-price over nested [t,t+h]. Vectors =
  ONE XGBRegressor(multi_output_tree, binary:logistic) over stacked soft-two-hot labels; RET averaged over horizons.
- **Result (rankIC / net SR / DD / turn):** DM .0493/**.69**/−20.8/3.8 ; TVAL(h6) .0535/**.69**/−21.8/3.1 ;
  tval_2 .0498/.65/−22.5/3.8 ; tval_2·3 .0497/.65/−22.1/3.7 ; tval_2·3·4 .0494/.65/−22.3/3.6 ;
  tval_2·3·4·5 .0505/.66/−22.5/3.4.
- **Conclusion:** (1) **NO** — every short-horizon vector (.65–.66) sits BELOW single TVAL(h6) (.69). (2) **Never
  climbs** — 2/2·3/2·3·4/2·3·4·5 = .65/.65/.65/.66, flat; short horizons near-redundant, worse DD, higher turnover.
  Confirms the long-end result (blending {3,6,12} didn't beat best single) now at the short end too. (3) **DM ≈
  TVAL(h6) — TIE on SR**; TVAL wins the capacity tiebreakers (higher IC + lowest turnover 3.1). **HONEST
  CORRECTION:** the big TVAL≫DM gap in T19 (.65 vs .40) does NOT reproduce on top-1000 — that gap was
  FULL-UNIVERSE/small-cap; on the tradable liquid set the target advantage collapses to a turnover/IC refinement.
  **Verdict: a single 6-month horizon is the entire edge** (as tval, slightly cleaner, or as DM's return-decile,
  equivalent). Multi-horizon short-window vector = REJECTED (complexity with no SR).

### T19 — 2026-07-17 · FULL-UNIVERSE confirmation of the enhancement stack (relative) — ★ holds up
- **Q:** does the stack (classify → full histogram → tval target) hold on the FULL universe via the real engine?
- **Method:** `research/confirm_tval.py` FULL liquid universe (~2687 names), long-decile net, seeds=1, RET=Σpₖμₖ
  ranking. Arms: resmom (naive) / dm_2bucket (P(top)-P(bot)) / ret_hist (10-bucket RET) / tval_hist (10-bucket,
  tval target). CAVEAT: long-only-decile screen, seeds=1, COMPACT features — NOT DM.py's L/S-5%/Han/seeds=5
  pipeline, so absolute levels are NOT the DM-1.11 baseline; RELATIVE deltas only.
- **Result (rankIC / net SR / DD / turn):** resmom .0334/.56/−21/2.3 ; dm_2bucket .0631/.40/−31/5.0 ;
  ret_hist .0795/.52/−25/3.3 ; **tval_hist .0913/.65/−23/2.5**.
- **Conclusion:** **MONOTONE confirmation at full scale.** Each thesis step adds IC: classify (dm2) > naive;
  full 10-bucket RET (ret_hist) > 2-bucket AND smoother (turn 5.0→3.3, DD −31→−25 = "RET is a smoother mean");
  tval target (tval_hist) > return target. **tval_hist wins net SR (.65) at LOWEST turnover (2.5) + good DD** —
  the full stack beats the naive floor net. Enhancement holds up (relative). OPEN GATE: the ABSOLUTE "beats DM
  1.11" needs the FAITHFUL DM.py L/S-5%/Han/seeds=5 run (TARGET=ret vs tval) — DM.py honest seeds=1 was −0.32
  (T7), so first pin which config reproduces ~1.0 before any "beats DM" claim.

### T18 — 2026-07-17 · Stack the two denoisers? residual-tval vs tval · REJECTED (they're substitutes)
- **Q:** does slope/SE (significance-weighting) computed on RESIDUAL returns beat plain tval (total, 0.66)? (stack
  the two proven denoisers). `research/resid_tval.py`, liquid top-1000, long-decile net.
- **Result:** resmom .57 ; **tval_total .66** ; tval_resid .62 (rankIC .0154/.0191/.0171). Cross-sec corr
  tval_resid~tval_total **0.93** (substitutes). Holding tval fixed, residualizing HURTS (.66→.62).
- **Conclusion:** **REJECT stacking — the two denoisers are SUBSTITUTES not complements.** Significance-weighting
  (SE denominator) is the STRONGER denoiser (already isolates the clean trend); residualizing on top adds noise
  (noisy daily β — same fragility as T14 multi-factor residual). Reconciles T14 (residual won on histogram-IC
  harness) vs here (significance-weighting wins on net-SR): different harnesses, and significance-weighting
  dominates for the deployable book. **LOCKS IN: MOM signal = tval (slope/SE, TOTAL), net SR 0.66** — champion
  across 4 harnesses (T15 ICIR .446, ml_tval .67, T17 .66, here .66). Vector-target variable = forward TVAL-TOTAL
  (not residual-tval). MOM sleeve essentially SETTLED: multi-horizon histogram over forward tval-total.

### DENOISING reframe (2026-07-17) — the unifying lens for targets
The forward return = drift + noise; point-return targets predict the NOISY realized value. **Trend-scanning's
IDEA (predict the denoised drift) is correct — its FLAW is the horizon SELECTION (look-ahead) + overlap, not the
denoising.** So most target improvements are DENOISERS of different noise sources: residual=factor noise,
vol-scale=heteroskedastic noise, HL-Gauss=label-discretization noise, cumulative=idiosyncratic-month noise,
fixed-horizon trend-slope=path noise (the HONEST trend-scan, no window selection), smoothed-endpoint=endpoint
noise. Denoising = raising target SNR = the lever ([[target-snr-return-vs-secondmoment]]).

### T10 — 2026-07-17 · TARGET sweep (histogram rep, DM full-univ harness, seeds=1) — ★ RESIDUAL wins
- **Hypothesis:** denoised targets beat the raw single-month return target on IC. Baseline = REP=hist raw/hard/K10
  = IC .0378 (T7); regular DM (2-bucket) = .0359 (T7).
- **Method:** `DM.py` REP=hist, one axis at a time: resid(RESID=1); voladj(VOLADJ=1). Metric: IC.
- **Result:** **resid IC .0439** (+.0061 vs hist-raw, +.0080 vs regular DM) ; voladj .0349 (−.0029, worse).
- **Conclusion:** **ADOPT residual target; REJECT vol-scaling.** Factor-neutralizing (Blitz) is the highest-
  leverage target change — the only denoiser that helped. Vol-scale adds more estimation noise than it removes
  (cross-sectional ranking already handles scale). Confirms the report's #1 call.

### T11 — 2026-07-17 · CREATIVE denoised targets — the honest trend-scan · REJECTED
- **Hypothesis:** the honest trend-scan (fixed-horizon forward OLS SLOPE of log-price — denoised drift, NO window
  selection, structural flaw removed) beats the raw return target.
- **Method:** `DM.py` REP=hist TARGET=slope, full univ, seeds=1. Metric: IC.
- **Result:** slope IC **.0350** (−.0028 vs hist-raw .0378, ~tied with regular DM .0359).
- **Conclusion:** **REJECT — trend-scanning's idea NOT vindicated even de-flawed.** Removing the look-ahead
  selection was not enough: the linear-SLOPE target itself DISCARDS signal (over 1–3mo it keeps only the slope,
  throws away the endpoint level that carries cross-sectional rank). Denoising via linear fit removes signal, not
  just noise, here. Plain (residual) return is the better target. (smooth/slope_hlg not run — batch cut for speed.)

### UNTESTED target axis (deferred): HL-Gauss SOFT bins (SMOOTH>0, K=50–100)
Killed for speed (K=50 full-univ = ~13min/run, too slow). The report says Gaussian-smoothed > hard two-hot
(Stop-Regressing). RUN ON THE FAST HARNESS (hist_mh.py, top-1000, ~2–3min) — full-universe DM.py is the WRONG
harness for target screening (protocol: screen on top-1000, confirm winners on full).

### T14 — 2026-07-17 · Residual predictability & noise (fast harness, top-1000) — ★ market-residual cleaner
- **Question:** how predictable/noisy is residual vs total, and does multi-factor residual enhance?
- **Method:** `research/resid_predict.py` top-1000, histogram model, per-month rank-IC DISTRIBUTION vs realized
  TOTAL fwd return. Targets: total; res1 (market residual); resM (market+size+mom+val residual). Metrics: mean IC,
  IC std, ICIR, t-stat, %pos, IC autocorr(1).
- **Result:** total meanIC .0256/ICIR .22/t 2.40/%pos 65 ; **res1 .0347/ICIR .28/t 3.05/%pos 64 (cleaner+stronger)** ;
  **resM −.0137/t −1.00/%pos 45 (BROKEN)**. IC autocorr(1) ≈ **0.78** for all.
- **Conclusion:** (1) **market-residual is a strictly better target** — higher IC, t-stat, ICIR (confirms T10).
  (2) The edge is REAL (t 3.0, 64% +months) but LOW-SNR & noisy per-month (IC std ~3.6× mean) — normal for
  cross-sec equity; monetized by breadth. (3) **IC autocorr 0.78 = the signal is REGIME-PERSISTENT** → hook for
  STATE-conditional sizing (size up when recent IC high). (4) **resM negative = do NOT residualize against the
  MOMENTUM factor** (strips the alpha) — over-orthogonalization confirmed empirically (= the T13 caveat). Correct
  multi-factor residual = market+size+value EXCLUDING momentum. NEXT: test mkt+size+val residual (no mom).

### T15 — 2026-07-17 · DENOISE the target by SIGNIFICANCE (trend-scan's real mechanism) — ★ big win
- **Question:** T11 rejected trend-scan's SLOPE — but that dropped the SE denominator that IS the denoiser. Does
  SIGNIFICANCE-weighting the target (weight each fwd move by signal/noise, so a clean +15% ≠ a jagged +15%)
  denoise? All arms = SAME histogram (K=10 softmax CE) method; only the TARGET changes.
- **Method:** `research/denoise_target.py` top-1000, per-month rank-IC + ICIR vs realized TOTAL fwd return. Targets:
  total; slope (magnitude only); **tval=slope/SE** (trend-scan t-value); r2ret=ret×R²; studz=ret/path-std;
  res1 (market-residual, reference).
- **Result (meanIC / ICIR):** total .0450/.373 ; slope .0420/.360 ; **tval .0641/.446** ; studz .0617/.429 ;
  r2ret .0542/.404 ; res1 .0546/.386. (t-stats 4.8–6.0; %pos 65–67.)
- **Conclusion:** **ADOPT significance-weighting — the denoising is the DENOMINATOR.** tval (slope/SE) lifts IC
  +42% (.045→.064) and ICIR +20% (.373→.446); slope (no denominator) is WORST → confirms T11 failed only because
  I dropped the SE. tval > res1 → path-noise denoising beats factor-residual denoising, and they attack DIFFERENT
  noise (path vs common-factor) → STACK them next. Why ICIR rises: denoised target has higher SNR → model fits the
  PERSISTENT signal not month-noise (IC autocorr .78) → edge bigger AND in more months; ICIR is the deployable
  metric (IR=IC√breadth). NOTE: IC std actually ROSE (trend-significance is more regime-dependent → more reason to
  state-condition sizing). Denoising ⟂ the histogram method (loss); they compound. NEXT (T17): residual × tval stacked.

### T16 — 2026-07-17 · NO-TARGET heuristic vs SUPERVISED, net of costs [RUNNING]
- **Question (user):** what if we use NO target — just rank what's working and buy? Does supervised beat the
  heuristic NET of costs? (Lit: much of ML's edge collapses to "recent winners continue"; ML's real value = cost-
  awareness/combination, not raw prediction; naive daily "buy today's winners" = short-term REVERSAL + overtrading
  = loses; momentum needs medium formation horizon.)
- **Method:** `research/naive_vs_ml.py` top-1000: naive_ret12 / naive_resmom (rank raw signal, NO model) vs
  ml_total / ml_tval (histogram). Metric: rank-IC + top-decile long NET SR / maxDD / turnover.
- **Result:** naive_ret12 .52 / naive_resmom **.58** / ml_total .56 / ml_tval **.67** (net SR; DD −22 to −27%;
  turn: naive 2.5, ml_total 4.0, ml_tval 3.0).
- **Conclusion:** **The ML machinery ALONE does NOT beat the free heuristic** — naive_resmom (0.58, no model,
  turn 2.5) BEATS the histogram on a plain target (ml_total 0.56, turn 4.0). **ML earns its keep ONLY via the
  DENOISED target:** ml_tval 0.67 > naive 0.58 (+0.09 net SR). So the entire value-add is TARGET ENGINEERING
  (significance-weighting/denoising), not the model — matches the literature (ML's edge = target/cost, not raw
  prediction). Deployable floor = residual-momentum RANK (cheap, 0.58); denoised-momentum tops ~0.67. Both are
  MOMENTUM (tval couples direction). The orthogonal STRENGTH leg (adds ON TOP?) remains the open T17 gate.

### External paper integrated (2026-07-17) — Schmerling "Slope, Strength, and Retail Extrapolation" (UChicago Booth, working paper)
Regress CUMULATIVE return on time (Sₜ~1+t) → decompose equity curve into SLOPE β (drift rate, center-weighted
hump filter) + unsigned R² (trend STRENGTH/smoothness). Conditional 5×5 sort **R²-FIRST then slope-within**,
long smooth-uptrend / short noisy-downtrend = TS factor, 4.6% FF6α (t=2.18) top500, 7.9% (t=3.44) full VW,
1962-2025. **RECONCILES OUR T11/T15:** our tval=slope/SE WON on rank-IC (T15) but the paper shows t(β)=slope/SE
earns ZERO FF6 alpha (βMom≈0.79, ABSORBED by momentum) — because IC-vs-total-return rewards a cleaner MOMENTUM
signal, while alpha measures UMD-orthogonality. So significance-weighting (tval/r2ret) = better MOMENTUM ranking
(keep for the sleeve) but NOT a new premium; the UMD-orthogonal alpha is UNSIGNED R² (strength) conditioned
STRENGTH-FIRST — any scalar COUPLING strength+direction (signed R², t(β), z·z) is absorbed by momentum.
**VALUE:** orthogonal to RESIDUAL momentum (Blitz) — spanning β=−0.05, α unchanged → STACKS with our T10/T14
champion = the orthogonal diversifier we want. **CAVEAT (big for us):** premium concentrates in LOW-IO/retail/
SMALL stocks (11.9% low-IO vs 0.07% high-IO tercile), survives costs only to ~10bps (short turns 80%/rebal) —
our LIQUID/large universe is exactly where it's WEAKEST → capacity/deployability risk. Mitigations: long-leg
RQ5/SQ5 alone = +8-11% single-decile alpha (fits long+exclusion, skips expensive short); paper is un-peer-reviewed.
Other: R² ⟂ Hurst/VR/autocorr (max|ρ|.11); formation inverted-U (neg@1mo reversal, peak@12mo, reverse@96mo);
mechanism = retail extrapolation from smooth charts (institutions tilt AWAY t=−4.23); alpha in stock SELECTION.

### T17 — 2026-07-17 · [PLANNED] Slope+Strength (R²-first double sort) on OUR liquid universe
- **Hypothesis:** (1) unsigned-R² STRENGTH earns a premium that the coupled t(β) does NOT (reproduce the paper's
  key distinction on our data); (2) it's ORTHOGONAL to residual momentum (the diversification claim); (3) it
  SURVIVES in the liquid/large universe we trade (the capacity gate — the paper's premium is in low-IO/small).
- **Method:** per stock, fit Sₜ~1+t on cumulative return over [t−252,t−21] → slope β, unsigned R². Conditional
  5×5 R²-first / slope-within, long RQ5/SQ5 (+ long-leg-only variant); vs univariate t(β) and unsigned-R². Net
  SR/DD + spanning vs res1 (T14 residual champion) + IO/liquidity tercile split. seeds=1 screen then confirm.
- **Method run:** `research/strength_resmom.py` — FACTOR SORT (no model): strength=unsigned R² of cumulative-return
  regression Sₜ~1+t over [t-252,t-21]; resmom champion; tval=slope/SE. Long-decile net + strength-gate + 3 gates.
- **Result:** resmom_D10 IC .0156/SR .57 ; **strength_D10 IC .0059/SR .64** ; tval_D10 IC .0194/SR .66 ; slope .0107/.53 ;
  resmom|strength-gate SR .57 (corr **+1.00** w/ resmom). resmom-IC by strength tercile: low .0129 / mid .0156 / high .0104.
- **Conclusion:** **REJECT for our universe — strength does NOT survive in LIQUID (all 3 gates fail).** (1) strength
  IC≈noise (.006) → its SR .64 is a QUALITY/low-vol LEVEL tilt, not cross-sec alpha; (2) gating resmom by strength =
  IDENTICAL to resmom (corr +1.00, same SR) → zero diversification; (3) NO strength×direction interaction (resmom
  flat/worse in high-strength). **Confirms the capacity caveat: Schmerling's premium is low-IO/small; GONE in liquid/
  large where WE trade.** So "trend + resmom" = just RESMOM for us; best single = denoised momentum (tval .66 = ml_tval
  .67). VECTOR-TARGET IMPLICATION: strength leg earns no slot in liquid → deployable vector target = MULTI-HORIZON
  BIMODAL RESIDUAL-RETURN histogram (hist/histMH), NOT [hist+strength]. (Tested strength as a FACTOR; a fwd-strength
  distributional TARGET is a different object, unlikely to help given IC≈0.)

### T13 — 2026-07-17 · [PLANNED] TARGET-SEPARATED sleeves (orthogonal / horizon-separated)
- **Idea (user):** deconstruct by sleeve so the 3 sleeves are TARGET-separated, not just feature-separated. Two
  forms: (1) HORIZON — reversion=short, momentum=medium, value=long (matches the return term-structure); (2)
  ORTHOGONAL — each sleeve predicts return residualized against the OTHER sleeves' signals (extends the residual
  win → diversified by construction). Compose: predict, at native horizon, the return-component orthogonal to the
  others = one decomposed conditional-return distribution.
- **Caveats:** don't OVER-orthogonalize (value×momentum interaction is a diversification FEATURE — residualize vs
  FACTORS is safe, vs each OTHER may strip alpha); output-blend already gives diversification (must beat it); the
  decomposition is itself a model (estimation error).
- **Method:** momentum histogram target residualized vs value+reversion signals; measure COMBINED-portfolio SR +
  cross-sleeve corr vs current blend. Adopt only if corr drops WITHOUT cutting standalone edge.
- **Result / Conclusion:** pending.

---

## REJECTED (do not re-test) — from this session and prior
- Quantile-grid target (T1); mixture-density NLL (T2); squeeze-aware read-off for the LONG book (T6).
- HMM/GMM/jump regimes (state layer); symmetric shorting of losers (short = tax; use exclusion).
- Trend-scan as a *separate* short target — the histogram subsumes clean-vs-squeeze as tail-shape read-offs.
- NESTED / overlapping multi-horizon tval vectors {2,2·3,2·3·4,2·3·4·5} (T20) — never beats single 6m; the horizons
  are collinear (adj corr .78–.94) so it's near-mechanical. AMENDED by T21: multi-horizon is NOT dead in general —
  DISJOINT-tval (non-overlapping windows) DOES beat single 6m on DD (.72/−20.2 vs .71/−23.0). The rule: MH needs pieces
  that are BOTH independent AND denoised. Still rejected specifically: nested/overlapping tval, and disjoint-RETURN
  segments (independent but undenoised = worst arms .43–.49).

## OPEN QUESTIONS
- T7 result (hist vs regular DM) + clean multi-horizon isolation (H=1 vs H=1,2,3 within histogram).
- HL-Gauss soft-bins (100 bins, σ=2·width) vs hard buckets; RESIDUAL-return target (Blitz).
- Cross-sectional attention trunk with the histogram/CE head (joint representation across names) — the frontier.
- Seeds=5 / full-universe / real-engine confirmation of every screen winner before ADOPT.

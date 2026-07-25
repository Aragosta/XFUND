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

## HAN'S BIMODAL DISTRIBUTION — how Han uses it, whether WE capture it, and where the tail belongs (2026-07-24, re-read pp.1-8)

**HOW HAN USES THE BIMODALITY (the actual paper, not the summary):**
- **The fact (Fig 1):** conditional on momentum, forward-return cross-sec distributions are U-shaped BIMODAL — a past
  **winner** is most likely high-return but **2nd-most-likely a crash**; a past **loser** is most likely low-return but has
  **significant squeeze mass**. "The bimodality is more evident among past losers." → momentum is *fundamentally* risky and
  the MEAN sits in the empty valley, so point/argmax prediction fails.
- **DM is a METHODOLOGY, not a target (Han p.4, his words):** *"not about finding new features but developing a new
  methodology that utilizes nonlinear information from existing features."* The method: classify → predict the full
  return-class **distribution** → **RECLASSIFY** `RET = Σ pₖμₖ` (law of total expectation; or Sharpe for short-constrained)
  → rank. Reclassification does three jobs: (a) fixes bimodality-poisons-the-mean, (b) fixes ML's data-imbalance (too many
  names dumped in the bottom class), (c) **pushes bimodal/risky names OUT of the extremes** (short-book avg size $153M →
  $1,589M). The objective-misalignment Han fixes (p.6): *"a classifier predicts the most PROBABLE class; an investor wants
  the EXPECTED return"* — his fix is the reclassification, NOT a Sharpe loss (evidence our objective isn't the bottleneck).
- **WHY momentum survives where others fail (Han's opening, p.1):** momentum is *"perhaps the most persistent anomaly …
  while the majority of firm characteristics have been revealed insignificant (Green 2017; Freyberger 2020), price momentum
  remains significant"* and is the top ML feature (Gu-Kelly-Xiu). Freyberger: only **~10** characteristics carry real
  incremental power — momentum is one survivor (also value/quality/investment/low-vol). Momentum survives = behavioral
  premium that doesn't arb away + cleanest data (no restatement/survivorship) + the nonlinear bimodal richness. **MOM is not
  the only real premium — it is the strongest/cleanest of ~10; breadth comes from the OTHER survivors, not more momentum.**

**MOM = DM's SAME estimator, different target (confirmed):** we apply Han's exact reclassification methodology to the **tval**
target instead of the **return**-decile target. → books 0.89-0.90 correlated (Stage 2 / T30) → ONE object; keep tval (T22).

**DO WE CAPTURE THE FULL BIMODAL STRUCTURE? — NO, by design (and it's tested):**
- ✅ We use the reclassification MACHINERY (histogram → RET), which beat point-regression (T5/T19).
- ✗ We predict the distribution over **TVAL, not RETURN.** tval's SE-denominator already penalizes jagged/squeeze-prone paths
  → tval is **pre-tamed**; we tame the bimodality *in the target* rather than model it in return-space.
- ✗ We **discard the shape** (rank on the reclassified mean only). Reading tail-mass as a directional crash/squeeze signal
  **FAILED** (T23 shape-not-tradable; T30 return-reclassification cosmetic) — only the mean ranks.
- ✗ The short-side squeeze machinery (Han's biggest win) is **MOOT** for us — MOM is long-only, so we *decline* the squeeze
  by not shorting. Our long book eats only the **crash** mode on winners.
- Our crash protection therefore comes from: the **tval target** (SE = squeeze/jaggedness filter) + **reclassification**
  (robust mean) + **CONSTRUCT liquidity-weighting** (down-weights the volatile small-caps where bimodality is most extreme).

**HOW WE MIGHT USE IT (the tail-risk plan — the bimodal tail is a RISK object, not a directional signal):**
Reading the tail as *alpha* is dead (T23). Using it as a **direction-aware RISK veto** is proven-adjacent
([[short-vol-thesis-proven]]: capping squeeze flipped a short book SR 0.28→1.69; corr(short PnL, squeeze rate) −0.76). Split
by scope, same as liquidity:
- **Per-name, direction-aware tail veto → shared CONSTRUCT layer.** LONG a name → down-weight high **crash**-propensity;
  SHORT a name → down-weight high **squeeze**-propensity. One shared weight, sign-flipped by leg. Sourced from the proven
  predictors (ivol / upvol / MAX5 / amihud / hi52 — [[asymmetric-feature-map]]) and/or the model's predicted-distribution
  **SPREAD** (2nd moment, ~2700× more predictable than the mean — [[target-snr-return-vs-secondmoment]]): the model *produces*
  the risk number, CONSTRUCT *acts* on it. This is the correct home for all the squeeze research — a construction VETO, not a
  failed second short-model ([[two-model-short-target]] concluded exactly "exclusion, not a full short book").
- **Aggregate squeeze/crash REGIME → STATE** (already partial: VIX→squeeze t+8.5, state_pc1 flags both — [[state-layer]]).
- **Value ranking:** highest on **MR** (trades BOTH bimodal legs — see MR_research.md) > MOM short (dead) > MOM long-only
  (crash-veto refinement; CONSTRUCT liquidity-weighting already absorbs much of it → the "may be cosmetic" arm).
PLANNED TEST (MECE, one axis): shared tail-veto in CONSTRUCT, A/B veto-vs-no-veto on MR first, then MOM long book.

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

### T40 — 2026-07-24 · TWO-MODEL asymmetric momentum (dedicated short head) — ★ NO short head crosses zero gross; squeeze-aware best but momentum has no short premium (4th confirmation)
- **Q (user "try a separate short & long model into a combined book"):** does a DEDICATED short model (own target) beat shorting the long
  model's bottom-decile? DECISIVE metric = each short head's STANDALONE GROSS SR (T39: the short leg's disease is ~zero gross alpha).
- **Method:** `research/mom_research.py` `_twomodel` (STAGES=TWO). Long head = tval champion. Short heads: (a) tval [shared model baseline]
  (b) single_hist("fwd") 1m-return reclassification (c) dm_2bucket("fwd") squeeze-aware P(bot1m)−P(top1m), short lowest = decline w/o
  squeeze. Combined = long-tval + short-head bottom-decile, dollar-neutral borrowable. top-1000 seeds=1 net. Reports combined + short-ONLY gross/net SR.
- **Result (combined SR / DD | short-only gross SR / gross ann):** long-only 0.43 ; short=tval −0.22/−85% | **−0.22**/−7.0% ;
  short=ret1 −0.21/−78% | **−0.19**/−4.8% ; **short=sqz-aware −0.17/−77% | −0.16/−4.5%**.
- **★ Conclusion — two-model does NOT fix it; NO dedicated short head produces POSITIVE gross short alpha.** The squeeze-aware target is
  the BEST short head (short gross −0.22→−0.16, combined −0.22→−0.17, DD −85→−77 — a squeeze-avoiding dedicated short helps at the MARGIN,
  matches [[two-model-short-target]] "modest") but CANNOT cross zero. Momentum FEATURES don't contain a "losers underperform" signal —
  losers DON'T underperform (Daniel-Moskowitz). 4th independent confirmation there's no short-momentum premium (T35 veto, T37 reclassif,
  T39 decomp, T40 two-model). Market-neutral momentum from the momentum signal alone is structurally ≤0. To go market-neutral: short leg
  must carry a DIFFERENT premium (value/quality/short-interest), or long-only + orthogonal sleeve. GATE: seeds=1 top-1000.

### T39 — 2026-07-24 · FULL L/S ANATOMY (leg×cost decomp + RISK action ladder + liquidity×squeeze) — ★ the short leg has ~ZERO gross alpha then pays 4% cost; it's economic (Daniel-Moskowitz), not a RISK failure
- **Q (user):** why is the L/S book not producing? what is RISK really doing? analyze the liquidity × squeeze interaction.
- **Method:** `research/mom_research.py` `_decomp` (STAGES=DECOMP). A: split the REAL DM book (/tmp/dm_weights.pkl, full univ) into
  long/short legs, attribute gross → −tc → −borrow → net. B: RISK action ladder on MOM(tval) L/S (equal→mdv→band→veto→beta-neut), net+
  gross+turn. C: corr(log-size, squeeze-propensity) among shorted names. top-1000 (B/C), full (A), seeds=1.
- **★ Result PART A (leg × cost, real DM full univ):** LONG gross SR 0.50/+5.8% → net 0.31/+3.2% (tc −2.6%). SHORT **gross SR −0.01/−2.6%**
  → net −0.22/**−6.6%** (tc −1.9%, borrow −2.2%). COMBINED gross 0.36/+4.8% (long carries) → net −0.03/−2.0%. **The short leg has ~ZERO
  GROSS ALPHA before any cost, then pays 4.1%/yr (tc+borrow) → −6.6% net, dragging the good long leg (+3.2%) to −2.0%.** The +0.041 combined
  IC (T38) was LONG-DRIVEN — winners outperform, losers DON'T underperform (Daniel-Moskowitz: a loser = "short a call", +E[r] in most
  states, crashes only in bear-rebound). NO short-side momentum PREMIUM exists.
- **Result PART B (RISK ladder, net/gross/turn):** equal −0.22/+0.07 ; +mdv −0.21/−0.06 ; +band −0.22/−0.08 (turn 7.1→5.9) ; +tail-veto
  −0.30/−0.15 (HURTS) ; **+beta-neut −0.15/+0.03 (only action that helps** — strips crash beta, DD −85→−77). mdv/band are LONG-book tools;
  veto is the wrong tool (redundant w/ reclassification); beta-neut helps but can't make zero alpha positive.
- **Result PART C (liquidity × squeeze):** corr(log-size, squeeze-prop) among shorted names = **−0.02 (≈ZERO)** → size & squeeze are
  SEPARATE non-overlapping axes. The borrow filter (mdv>$25M) removes microcap squeezers UPFRONT; within liquid names squeeze is
  idiosyncratic → mdv-weighting CAN'T address it → squeeze MUST be handled in the ALPHA (reclassification), not by liquidity sizing.
- **★★ Conclusion — the L/S loses because it bolts a ZERO-ALPHA, 4%-COST leg onto a good long leg; this is ECONOMIC, not a RISK bug.** The
  RISK layer did everything it can: the reclassification KILLED the squeeze (gross short went from −85%-DD catastrophe to a controlled
  −0.01 SR — the fix WORKS), beta-neut cut the crash. But momentum has NO short-side premium (Daniel-Moskowitz) — RISK can neutralize a
  risk, it can't create a missing alpha. **Market-neutral momentum from the momentum signal alone ≈ 0.** For a positive market-neutral
  book the short leg must carry a DIFFERENT premium (short expensive/low-quality/high-short-interest), or pair long-only mom with an
  orthogonal sleeve. Corrects T38's "cost problem" → more precisely "NO short alpha + cost" (≈half each of the −6.6%). GATE: seeds=1.

### T38 — 2026-07-24 · REAL DM sleeve, FULL universe, dollar-neutral, net (the deployable market-neutral number) — ★ reclassification RESCUES short (−0.22→−0.03) but net L/S is break-even; +IC −net = a COST/turnover problem
- **Q (user):** what's the honest DEPLOYABLE dollar-neutral number for the squeeze-protected book (real DM.py, multi-horizon, full univ)?
- **Method:** `SEEDS=1 LONGONLY=0 MINDVPCT=0 MINDVABS=5e6 python3 DM.py` — the actual production DM sleeve (MH-DM 1,2,3, return
  reclassification, Han+TS features), full liquid universe, L/S dollar-neutral (TOP_Q=0.05), net via BACKTEST.py.
- **Result:** MH-DM (1,2,3): **IC +0.041 · net SR −0.03 · ann −2.0% · maxDD −72.6% · turn 8.5.**
- **★ Conclusion — mechanism VINDICATED, but the deployable dollar-neutral book is BREAK-EVEN; it's a COST problem now.** The
  reclassification rescues the short from catastrophe (tval-short −0.22 → full MH-DM −0.03) — user's bimodal thesis confirmed on the real
  sleeve, full universe. BUT ★ **IC is POSITIVE (+0.041) while net is NEGATIVE (−0.03)** → the short leg ranks correctly and is gross-
  positive, but its turnover (8.5 vs long-only 3.4) + borrow eat the edge. The squeeze-protection neutralizes the short leg's DAMAGE but
  the residual short alpha is too thin to pay its own cost. So the honest market-neutral momentum book ≈ 0 net; long-only (0.43 top-1000,
  ~0.58 full) stays the deployable book. LEVERS to tip L/S positive (ranked): (a) short-leg TURNOVER reduction (banding/holding/cost-in-
  objective = the parked Pelger lever), (b) beta-neut (screen −0.22→−0.15), (c) pair long-mom with a DIFFERENT short-premium. NOTE: the old
  "DM-native L/S ~1.0-1.36" memory numbers were pre-[[data-substrate-upgrade-2026]] / directional-tilt / beta-neut, NOT this honest dollar-
  neutral net. GATE: seeds=1 full. NEXT: seeds=5 + short-leg turnover cut (band the short leg) + beta-neut on full to see if L/S clears 0.

### T37 — 2026-07-24 · MULTI-HORIZON reclassification short + tval TERM-STRUCTURE slope — ★ MH reclassification confirms/improves the squeeze-protected short (−0.03, DD −60%); term-structure slope REJECT
- **Q (user):** (1) the REAL DM is MULTI-horizon return reclassification (H=1,2,3) — does it beat single-horizon ret1 (T36)? (2) ★ user's
  idea: multi-horizon reveals the TERM STRUCTURE — predicted return RISING across horizons = accelerating (long), FALLING = rolling over
  (short); slope = RET@r3−RET@r1 as a directional (squeeze-protected) short signal.
- **Method:** `research/mom_research.py` `_mh_short` (STAGES=MHSHORT). `multi_hist(["r1","r2","r3"])` (single-month fwd returns h=1,2,3,
  ONE multi_output tree) → per-horizon RET. LVL=mean across horizons (MH-DM short); SLP=RET@r3−RET@r1 (term-structure). Long=tval. top-1000 seeds=1 net.
- **Result (net SR / maxDD / IC6m):** long-only(tval) 0.43 ; **L/S MH-DM both −0.03 / −59.9% / .045** ; L/S long-tval/short-MH-DM −0.10 ;
  **L/S term-structure slope both −0.48 / −88% / −.006** ; L/S long-tval/short-decel-slope −0.56 ; score-corr(level,slope)=+0.03.
- **★ Conclusion — MH reclassification CONFIRMS the squeeze-protected short + improves DD; term-structure slope REJECT.** MH-DM short
  −0.03 (vs single-horizon ret1 −0.04, T36) with DD −72%→−60% — multi-horizon denoises the drawdown (as it does for the DM long book).
  The reclassification-in-alpha squeeze filter is now validated 3 ways (T36 single, T37 MH). Still ~break-even at seeds=1/top-1000 →
  confirm the DEPLOYABLE dollar-neutral number on FULL universe seeds=5 via the real DM.py (LONGONLY=0). TERM-STRUCTURE SLOPE: REJECT —
  IC −0.006, books −0.48; differencing two noisy monthly-return reclassifications amplifies noise + trend-ACCELERATION (2nd derivative of
  price) is known-weak (echoes prior ACCEL-feature nulls). Orthogonal to level (+0.03) but orthogonal-and-empty. Legs must stay COHERENT
  (mixed long-tval/short-anything < both-coherent). GATE: seeds=1 top-1000.

### T36 — 2026-07-24 · Han RECLASSIFICATION as the squeeze filter — short on RETURN-reclassified score, not tval — ★ the reclassification WORKS: short −0.22→−0.04; the veto was the wrong tool all along
- **Q (user "losers bounce is EXACTLY what the bimodal dist addresses — start with the Han paper"):** does shorting on the RETURN-
  reclassified score (RET=Σpₖμₖ, which pulls bounce-prone losers OUT of the short extreme) rescue the leg that TVAL-short blows up?
- **Method:** `research/mom_research.py` `_han_short` (STAGES=HAN). Re-grounded in Han pp.1-8 (MOM_research.md L27: "tval predicts
  E[tval] not E[return] → LOSES short protection; return-RET L/S survives"). Long = tval; short scores = tval vs single_hist("fwd")=
  1-month RETURN reclassification vs single_hist("fwd6"). top-1000 seeds=1 net.
- **Result (net SR / maxDD):** long-only(tval) 0.43 ; L/S short=tval **−0.22 / −85%** ; L/S both=ret1 (pure Han DM) **−0.04 / −72%** ;
  L/S long-tval/short-ret1 (mixed legs) −0.21 ; L/S long-tval/short-ret6 −0.42.
- **★ Conclusion — the RECLASSIFICATION is the squeeze filter, and it WORKS (user vindicated).** Shorting on the 1-month RETURN-
  reclassified score instead of tval moves the short leg −0.22→−0.04 (DD −85→−72) — the bounce-prone losers get their RET pulled up by
  their upside mass and never enter the short book (Han's $153M→$1589M mechanism, on OUR data). This CORRECTS T34/T35: those shorted on
  TVAL (loses protection) + bolted on a VETO (wrong tool) — the working mechanism is baked into the ALPHA score (like the liquidity
  filter is baked into construction), NOT a RISK-layer veto. CAVEAT: legs must be COHERENT — mixed long-tval/short-ret1 fails (−0.21);
  the working book is Han's DM with BOTH legs return-reclassified (−0.04). Still slightly negative at seeds=1/top-1000, but Han's own L/S
  is +0.22 and DM-native memory shows ~1.0-1.36 L/S on FULL universe → MULTI-HORIZON (T37) + seeds=5 + full should push positive. This
  IS the DM sleeve. GATE: seeds=1 top-1000. See [[bimodal-tail-risk-construct]].

### T35 — 2026-07-24 · FULL RISK layer on L/S (mdv+band+tail-veto+beta-neut, fresh score) — ★ short leg's disease is momentum-CRASH BETA, not squeezes; veto aimed at wrong risk; still long-only
- **Q (user "solvable in this EXACT RISK setup"; stop-loss=META not RISK):** T34's veto used an EQUAL-WEIGHT hand-rolled short book —
  unfair (discarded mdv+band, the RISK layer's power). Does the REAL `RISK.risk_book` (SELECT→SIZE mdv→TAIL-VETO→beta-NEUT) salvage L/S?
- **Method:** `research/mom_research.py` `_risk_ls` (STAGES=RISKLS). Fresh champion score (single_hist tval, NOT stale Jul-18 parquet) →
  `RISK.risk_book(weighting=mdv, band=.10/.20, ls=True)` under veto-strength × beta-neut grid. top-1000 seeds=1 net.
- **Result (net SR / maxDD):** long-only(mdv+band) **0.58 / −30%** ; L/S no-veto **−0.22 / −91%** ; L/S tail-veto soft(lam.5,fl.2) −0.25 ;
  L/S tail-veto HARD(lam1,fl0) −0.30 ; **L/S beta-neut −0.15 / −77%** (best L/S) ; L/S tail-veto HARD+beta-neut −0.24.
- **★ Conclusion — the short leg's disease is momentum-CRASH BETA (systematic), NOT idiosyncratic squeezes.** 3rd faithful confirmation
  the tail-veto can't salvage L/S — now via the EXACT production RISK layer at full strength (mdv+band), so the "unfair equal-weight"
  objection is closed. The DIAGNOSTIC: only BETA-NEUT helps (−0.22→−0.15, DD −91→−77) → the whole short book (yesterday's losers) rips up
  TOGETHER on momentum reversal = correlated crash beta; a per-name squeeze-veto trims lottery names but leaves the systematic crash
  untouched AND concentrates the book (turn↑) → net worse. Even beta-neut leaves it NEGATIVE because shorting weak-trend names is a
  structurally bad SHORT (losers bounce, [[short-leg-is-the-tax]]). **The RISK layer cannot fix a bad short-ALPHA** — a profitable short
  needs a DIFFERENT signal (short overvalued/squeeze-prone names, not momentum losers) = an ALPHA/META decision, not a RISK action.
  MOM stays LONG-ONLY; market-neutrality is a PORTFOLIO job (pair with orthogonal premium). GATE: seeds=1 top-1000.

### T34 — 2026-07-24 · Bimodal tail-mass VETO — can the risk layer salvage the L/S short leg? — ★ NO; every veto form makes L/S worse → LONG-ONLY confirmed
- **Q (user "RISK layer was supposed to salvage L/S via bimodal protection"):** does a FORWARD-LOOKING squeeze forecast (return-
  histogram upper-tail mass) beat the realized semi-vol veto, and does HARD exclusion beat soft down-weight, at rescuing the short leg?
- **Method:** `research/mom_research.py` `_hist_veto` (STAGES=VETO). Ranking = MOM(tval) champion; short = borrowable bottom-decile;
  squeeze forecast = P(top-2 of 10 bins) from a fwd6-return softprob model (`hist_P`); semi-vol proxy = RISK.tail_props upside-semivol.
  Gross held at 1 per leg (dollar-neutral) so it's a pure composition test. top-1000, seeds=1, net.
- **Result (net SR / maxDD):** long-only **0.43 / −33%** ; L/S no-veto **−0.22 / −85%** ; L/S semi-vol down-wt −0.24 ; L/S hist-squeeze
  down-wt −0.26 ; L/S hist-squeeze HARD-exclude **−0.30 / −89%** (turn 7.1→7.5).
- **★ Conclusion — the bimodal veto CANNOT salvage L/S; every arm is worse than no-veto.** Forecast-histogram beat nothing; hard-exclude
  was the WORST. The short book's problem isn't the vetoable top squeezers — the WHOLE short-momentum book loses net (loser-bounce/
  momentum-crash + borrow), so excluding names just concentrates the remainder + raises turnover. Reconfirms [[target-universe-dependent]]
  / [[short-conditioning-and-state-timing]] with the strongest veto yet tried → **deploy LONG-ONLY.** NOT tested: payoff-space upside-cap
  / short stop-loss (the [[short-vol-thesis-proven]] 0.28→1.69 mechanism is a PnL cap, not name exclusion) — the only short lever left,
  but its realism (stops on a monthly book) is doubtful. RISK ACTION 3 tail-veto: keep OFF by default (flat on long, harmful on short).
  GATE: seeds=1 top-1000.

### T33 — 2026-07-24 · Factor momentum OPTIMAL-HARVEST sweep (window×#PC×horizon) + combine/subsumption — ★ strong⇔orthogonal is a TRADEOFF; no liquid config beats MOM; broad/anomaly-factor is the last lever
- **Q (user "one model per premium, optimally harvest it"):** find the factor-momentum config that best harvests the momentum
  premium; does it beat / absorb the MOM(tval) champion? (follow-up to T31.)
- **Method:** `research/mom_research.py` `_facmom_deep` (STAGES=FAC). PART A sweep win∈{36,60} × K∈{3,5,10,20} PCs; IC at native
  1/3/6/12m + long-decile net SR. PART B: best config ⊕ MOM(tval) z-sum combine + 50/50 blend + spanning regressions. top-1000, seeds=1.
- **Result — PART A (netSR / IC1 / IC12):** best = **win60 K20 → net SR 0.38** (IC1 +.005, IC12 −.009); more PCs HELP monotonically
  (K3 0.28→K20 0.38) and flip short-horizon IC positive — matches E-L "high-eigenvalue PCs carry it". win36 ≈ win60. All < MOM 0.43.
- **★ Result — PART B (the decisive one):** facmom(win60,K20) 0.38 · MOM(tval) 0.43 · **z-sum combine 0.44** (+0.01, DD −33→−27) ·
  50/50 blend 0.43. **book-corr(best facmom, MOM) = +0.74** — NOT T31's 0.15! SPANNING: MOM α|facmom +1.7%/yr (t=1.1), facmom α|MOM
  +1.1%/yr (t=0.5) — both INSIGNIFICANT, neither spans the other.
- **★ Conclusion — STRONG ⇔ ORTHOGONAL is a TRADEOFF; factor momentum does not give a strong-orthogonal harvest on liquid names.**
  T31's 0.15 orthogonality was a property of the WEAK low-K version; adding PCs to strengthen it (K20, 0.38) pulls corr to 0.74 =
  reconstructs stock momentum (E-L subsumption in reverse: high-eigenvalue factor momentum ≈ the momentum factor). No config is both
  strong AND orthogonal; none beats MOM(tval) 0.43; the combine adds +0.01. On TRADEABLE names, MOM(tval) already ≈ the optimal harvest.
  **The one remaining test of the premise = BROAD universe + ANOMALY-FACTOR PCs** (not raw-return PCA, not top-1000) — where E-L's edge
  actually lives (more names → cleaner high-eigenvalue structure). If it fails there too, factor momentum is closed as a MOM sleeve.
  GATE: seeds=1 top-1000 screen. See [[factor-momentum-orthogonal]].

### T32 — 2026-07-24 · Distributional readout A/B — mean-reg vs hist-CE (champion) vs HL-Gauss soft-bins — ★ hist-CE confirmed best; HL-Gauss REJECT
- **Q (user "test distributional"):** does the histogram cross-entropy readout beat point mean-regression, and do HL-Gauss *soft*
  (Gaussian-spread) bins beat the champion's hard two-hot histogram? (open Q from [[momentum-distributional-framework]].)
- **Method:** `research/mom_research.py` `_facmom_dist` PART B (STAGES=FAC). Same pool/features/tval-target, vary ONLY the readout.
  top-1000, seeds=1, long-decile, net. `mean_reg`=XGBRegressor on grank(tval); `single_hist`=softprob→RET (champion); `hlgauss_hist`
  =multi_output on Gaussian-spread soft labels (σ=1 bin) →RET.
- **Result (IC6m / net SR):** base mean-reg **.0343 / .39** ; hist-CE champion **.0460 / .43** ; HL-Gauss σ=1.0 **.0346 / .41**.
- **Conclusion — hist-CE (hard two-hot) stays champion.** Histogram-CE beats point mean-reg on BOTH IC (+.012) and SR (+.04) —
  reconfirms the distributional-loss lever ([[momentum-distributional-framework]]). HL-Gauss soft-bins do NOT add (.41<.43, below the
  +.10 bar; IC .035 ~ mean-reg's, i.e. the Gaussian smoothing washes the IC gain back out). REJECT HL-Gauss. Representation-vs-arch
  thesis holds: the *histogram* is the win, the *softness* is second-order. GATE: seeds=1 screen.

### T31 — 2026-07-24 · Factor momentum (Ehsani-Linnainmaa PC-timing) — orthogonality kill-test — ★ GENUINELY orthogonal (0.15) but weak standalone → NEEDS-MORE
- **Q (user "test factor"):** is PC-factor momentum the one lower-correlation momentum variant (per [[signals-vs-ml-momentum-redundancy]])?
  Build it, measure book-return corr to MOM(tval)/DM(ret). Bar: book-corr<0.60 → diversifier worth building; ≥0.80 → same bet, reject.
- **Method:** `research/mom_research.py` `facmom()` in `_facmom_dist` PART A. Per month: SVD trailing-60m return panel → top-8 PC
  loadings + factor returns; factor momentum = Σ past-11m (skip-1) factor return; stock score = Σ_k loading·factor-mom (PC-sign cancels).
  PIT/leak-free (window ends iloc[k-1]). top-1000, seeds=1, net. Dollar-neutral (market-neutral) + long-only; vs single_hist(tval)/(fwd6).
- **Result (IC6m / net SR / maxDD / turn):** facmom L/S **−.009 / −.17 / −87.6% / 6.1** ; facmom long **−.009 / .28 / −39.7% / 3.0** ;
  MOM(tval) L/S .046/−.22 ; DM(ret) L/S .010/−.48. **book-corr(facmom,MOM)=+0.15, (facmom,DM)=+0.25 ; score-corr(facmom,MOM)=+0.10.**
- **Conclusion — the orthogonality hypothesis is CONFIRMED, the standalone alpha is not (yet).** facmom is genuinely a DIFFERENT bet
  (0.15 book-corr ≪ 0.60 bar — vs the 0.74–0.90 that every price-momentum variant shows) → vindicates [[signals-vs-ml-momentum-redundancy]]
  that factor momentum is the ONE low-corr variant. BUT IC6m≈0 and standalone long-only only 0.28 in this naive form (raw-PCA, K=8,
  top-1000, evaluated at the 6m horizon). L/S −0.17 = the usual short-leg tax. **VERDICT: NEEDS-MORE** — orthogonality is worthless
  without alpha to combine; standalone-SR is the wrong gate for a diversifier (Protocol standing rule). Follow-ups, in order: (1) COMBINE
  test — does MOM ⊕ facmom raise COMBINED net SR despite weak standalone (the only test that matters)? back-of-envelope ≈0.45 vs MOM 0.43
  = marginal at this strength; (2) native horizon (factor mom predicts 1–12m, not 6m — evalfield mismatch may be suppressing IC);
  (3) proper E-L construction — anomaly-factor PCs (not raw-return PCA) + BROAD universe (factor mom concentrates in high-eigenvalue PCs,
  strongest broad) + time-series factor-timing, not forced pure-XS L/S. The signal is real and orthogonal; making it STRONG is a bigger build.

### T30 — 2026-07-24 · Unified multi-horizon tval model — MECE: 2-horizon readout + return-reclassification (static+rolling) — ★ both cosmetic/redundant; keep ONE simple readout
- **Q (user):** (idea 1) two-horizon readout from ONE model — does a short {1,2,3} readout blended with the 6mo tval offense add
  net? (idea 2) predict tval-hist but reclassify on RETURNS of the buckets — static AND rolling (crash-adaptive)?
- **Method:** `research/mom_research.py` `_unified_mh` (STAGES=UNI). ONE multi_output tree over [tval@6,r1,r2,r3]; read all arms off
  its bucket-probs. Construction FIXED = CONSTRUCT(mdv, band=.10/.20). Full univ ~3445, seeds=1, net. AXIS 1: O=tval@6 vs
  S=mean short-return readout{1,2,3} vs 50/50 blend (+corr). AXIS 2: gauss vs static-return-centers vs rolling(12m)-return-centers,
  overall + worst-10%-of-O crash months.
- **Result — AXIS 1 (net SR):** O tval@6 **0.70**/−25.3 ; S short{1,2,3} 0.59/−30.8 ; O+S blend **0.66** ; **corr(O,S)=+0.89.** FAILS both
  bars (corr≥0.8, blend<O). Short readout is a weaker 0.89-corr copy → blending DILUTES (0.70→0.66). Consistent w/ Stage 2 (0.90) + T20.
- **Result — AXIS 2 (net SR / crash-month/mo):** B0 gauss **0.70** / −4.90% ; B1 static-ret **0.70** (=B0 exactly, confirms T23 cosmetic) ;
  B2 rolling-ret12 0.69 / −4.91% (Δ vs B0 in crashes = −0.02%). FAILS — rolling reclassification is cosmetic overall AND in crashes;
  the tval→return map stays ~monotone so the rank doesn't move. Crash is a tail/construction problem, not a readout problem.
- **Conclusion — keep it SIMPLE: one model, one readout (tval@6), gaussian centers, CONSTRUCT(mdv,band).** (1) DM/short-horizon
  COLLAPSES into MOM (3rd confirmation of the ~0.9 redundancy; short readout dilutes). Caveat: Axis-1 S is a faithful-not-exact DM
  proxy → the definitive DM verdict is the real DM.py book vs MOM in ERC, but signal-level evidence is strongly against DM adding.
  (2) Return-reclassification (static+rolling) REJECTED — cosmetic (T23 reconfirmed; crash-adaptive falsified). Champion MOM book =
  tval@6 → gaussian RET → CONSTRUCT(mdv,band) = **net SR 0.70 / −25% DD / turn 2.3** (full univ, seeds=1). Session arc: equal-weight
  0.14 → construction 0.70; every model/readout elaboration cosmetic, construction was everything. GATE: seeds=1.

### T29 — 2026-07-24 · Path-to-unified-model, Stages 0-3 (construction, objective, DM+MOM unify, model-over-signals) — ★ CONSTRUCTION is the win; sophistication all fails
- **Q (user):** optimize for a different objective (direct-Sharpe/Stop-Regressing); combine DM+MOM into ONE model; get "one momentum
  model where output > sum of its signal inputs". Staged on trees before any DL (DL earned only if trees pay).
- **STAGE 0 — IC→SR leak (full univ ~3445, tval champion, vary ONLY construction):** ewfull IC.117/**SR .14**/−37.8/junk-frac **53%** ;
  liqrestrict(rank top-1000 only) .050/.42/−28.0/0% ; **mdvwt(liquidity-weight the decile) .117/.57/−28.9/53%**. ★ **The 0.14→0.57 leak
  is CONSTRUCTION, not signal/objective** — 53% of the equal-weight long-decile is illiquid small-caps; EW hands them full weight →
  vol/cost/delisting tank the book. mdvwt keeps full IC (.117) AND trades → dominates liqrestrict (which drops IC to .050 by discarding
  the broad-universe signal). LESSON: rank on EVERYTHING, SIZE DOWN small-caps (don't equal-weight, don't exclude). Production
  `mom_layer.backtest` equal-weights → ~+0.4 SR on the table. (top-1000 base ML_tval .43; full-univ EW .14 = the construction gap.)
- **STAGE 1 — objective (top-1000):** base_rankCE IC.046/SR **.43** ; top_focused(binary top-decile loss) .028/**.29**. ★ top-decile-focused
  objective HURT. Full-rank CE is fine; the objective was NOT the leak (Stage 0 = construction was). Cheap tree-proxy of "align the
  objective to the traded slice" FAILS → the direct-Sharpe DL motivation collapses (the IC→SR gap it targets is a construction bug).
- **STAGE 2 — unify DM+MOM as 2 readouts of ONE multi_output tree (top-1000):** MOM(tval) IC.042/SR **.47** ; DM(ret) .011/.28/−51.4 ;
  UNIFIED(rank-avg) .027/**.41** ; **corr(MOM book, DM book) = +0.90.** ★ Unifying makes it WORSE — DM is a weaker, 0.90-correlated copy of
  MOM; averaging dilutes the stronger tval readout (.47→.41). **DM is NOT a diversifier of MOM.** Collapse to one model, tval readout.
- **STAGE 3 — model OVER the signal library (+state) vs best-single & blend (top-1000):** best_single(tvalpast) IC.046/SR **.52** ;
  equal_blend .047/.49 ; model_signals **.051/.43** (highest IC, lowest SR) ; model_sig+state .051/**.43** (state adds 0.00). ★ "One model >
  sum of inputs" FALSIFIED on trees — the model overfits ranking (IC↑) but loses tradable SR, beating NEITHER best-single nor the blend;
  state-gating is inert. Confirms T26.
- **Conclusion — the alpha is CONSTRUCTION + orthogonal SLEEVES, not momentum-model sophistication.** Five levers now fail (ML vs raw T26,
  13F T28, objective S1, DM+MOM unify S2, model-over-signals S3); the ONE win is construction (S0, +0.43 SR). **ACTIONS:** (1) deploy
  liquidity/vol-aware weighting in mom_layer (rank-all, size-by-liquidity — the real free win, small-cap signal harvested by SIZING not
  exclusion); (2) collapse DM into MOM (one tval readout — DM-as-separate only dilutes + adds turnover); (3) FREEZE the signal/model; (4)
  DL direct-Sharpe = NOT justified (gate self-set: trees must pay first; S1/S3 negative). Budget → construction optimization + orthogonal
  premia/data, not momentum DL. GATE: seeds=1 screen (S0 full-univ, S1-3 top-1000).

### T28 — 2026-07-24 · crm/cmom (PIT 13F clusters) vs the FULL champion baseline + random control — ★ REJECT: weak-baseline artifact, no signal over random
- **Q:** the make-or-break test from [[asset-embeddings-sleeve]] — the crm(cluster-relative)/cmom(peer) momentum crash-cut
  (claimed net 0.56→0.74, maxDD −43%→−27%) showed vs a THIN baseline; does it SURVIVE the full 33-feat champion baseline,
  and beat a RANDOM-cluster control?
- **Method:** `research/mom_research.py` EXPERIMENT. PIT `clusters_ts_focused.parquet` (45d lag). crm{6,12}=mom−clustermean,
  cmom{6,12}=clustermean appended to champion X. Arms: BASE / BASE+EMB(real) / BASE+EMB(random shuffled labels). Full liquid
  (~3445 names), tval champion (single_hist), long-decile net, seeds=1, OOS 2011+. OMP_NUM_THREADS=8 (single-thread hung 12h prior).
- **Result (IC / net SR / maxDD):** BASE .1173/.14/−37.8 ; real .1201/**.22**/−40.5 ; random .1186/**.20**/−41.5.
  Δ(real−base) +.0028 IC / +0.08 SR / **−2.7% DD (WORSE)** ; Δ(random−base) +.0014 IC / +0.06 SR / −3.7% DD.
- **Conclusion — REJECT.** (1) real beats random by only +0.02 SR / +0.0014 IC = NOISE at seeds=1, below the +0.005 bar → the 13F
  co-holding structure adds ~nothing over SHUFFLED groupings; the "lift" is generic extra-feature nudging. (2) The −16pt crash-cut
  REVERSED: against the rich base, EMB makes DD WORSE (−37.8→−40.5), random too → the prior crash-defusing was a WEAK-BASELINE
  artifact (rich base already encodes peer/size structure). Resolves the yellow flag NEGATIVE. (3) **GNN NOT justified** — crm/cmom is
  the hand-coded message-passing proxy; it fails its own random control, so a graph-net generalization has no validated mechanism to
  inherit (protocol #1: don't chase complexity on an unproven mechanism). 13F co-holding axis = DEAD as a momentum improver in liquid US.
  Absolute BASE SR low (0.14) is harness (full-univ/seeds=1/tval/2011+) not the point — the random control is level-invariant. GATE: seeds=1.

### T27 — 2026-07-23 · "Uncrowded Momentum" (13F co-holding crowd-density) — ★ thesis FALSIFIED (backwards sign); confounded by look-ahead embedding
- **IDEA:** momentum crashes = crowded DELEVERAGING; the crowd is observable in 13F co-holding embedding space →
  long winners that are momentum-strong but LOW crowd-density (peripheral, un-arbitraged) should keep the premium with a
  smaller crash tail. A genuinely ORTHOGONAL (ownership-graph, not price) signal aimed at the crash.
- **Method:** `research/mom_research.py` EXPERIMENT. crowd-density_i = cosine(embedding_i, centroid of top-20% momentum
  winners), from `data/13f/embeddings_pooled.parquet` (2576×100). Books (long-decile, net, top-1000, 2011+): vanilla=z(mom);
  uncrowded=z(mom)−z(dens); crowded=z(mom)+z(dens); pure_uncrowded=−z(dens). CAVEAT: POOLED (static, look-ahead) embedding.
- **Result (IC / net SR / maxDD):** vanilla .023/.43/−33.4 ; **uncrowded .006/.38/−35.1** (WORSE SR AND DD) ;
  crowded **.033**/.42/−31.8 (higher IC, better DD) ; **pure_uncrowded −.061**/.27/−42.7 (NEGATIVE IC). Orthogonality
  corr(mom, crowd-density) = **+0.037** (genuinely distinct axis).
- **Conclusion — FALSIFIED as specified (sign is BACKWARDS).** Low crowd-density = NEGATIVE forward IC → orphan winners are
  junk; institutional co-holding reads as a QUALITY/durability endorsement, not fragility (in liquid large-caps). The one
  surviving positive: the axis IS ~orthogonal to price (+0.037). **BUT the verdict is CONFOUNDED:** the pooled embedding has
  look-ahead (a name that ENDED UP co-held = a survivor that succeeded → "crowded" ≈ "eventually-worked"), which plausibly
  manufactures the positive sign; and a static embedding cannot test the crash-time deleveraging mechanism (tail, not mean).
  **GNN NOT justified yet** (don't build graph infra on a proxy that showed the wrong sign — protocol #1 multiple-testing). Cheap
  discriminating next step (only if pursued): coarse PIT crowd-density from the 13F zips, judged on CRASH months + SHORT leg.
  Otherwise the tradable residue is a mild "crowd-CONFIRMED momentum" quality tilt (look-ahead-suspect). GATE: rule-based screen.

### T26 — 2026-07-23 · SIGNALS-vs-ML horse race + redundancy audit — ★ the whole momentum complex is ONE bet; ML doesn't beat a raw signal
- **Q (user):** should we drop per-sleeve ML and let a signal library (frog-in-pan, returns, …) + a disciplined combiner do the
  work — or do we need a better ML architecture? And how redundant are MOM/DM really?
- **Method:** `research/mom_research.py` EXPERIMENT block. 8 long-only decile books, IDENTICAL pool (top-1000, liquid, seeds=1,
  2011+, net engine): raw signals `mom11` (11m price mom), `resmom` (residual mom), `hi52` (52wk-high), `macd` (Baz composite),
  `tvalpast` (slope/SE of past-6m log price = the champion target computed BACKWARD as a free signal), `fip` (frog-in-the-pan =
  z(mom)−z(info-discreteness), Da-Gurun-Warachka, monthly-sign proxy); `composite` = equal-weight z-blend of all 6 (the
  disciplined "let the portfolio sort it out"); `ML_tval`/`ML_ret` = the actual XGBoost champion estimators (single_hist).
  Plus two redundancy matrices: book net-return corr + cross-sec score corr.
- **Result (net SR / IC / maxDD):** tvalpast **.52**/.046/−23.6 ; resmom .49/.036/−24.3 ; hi52 .49/.053/−29.2 ; macd .48/.037/−27.0 ;
  fip .48/.023/−28.0 ; composite .49/.047/−26.9 ; mom11 .43/.023/−33.4 ; **ML_tval .43**/.046/−32.7 ; ML_ret .30/.010/−49.1.
  **Book-return corr among the 6 raw signals = 0.86–0.99** (mom11↔fip .97, resmom↔mom11 .92); composite↔mom11 .97. **ML_tval
  score-corr to raw signals only .15–.42** (most distinct) but LOWER SR. **ML_tval↔ML_ret score-corr .70** (= the MOM↔DM
  redundancy, protocol's 0.74, from a 3rd angle).
- **Conclusion:** (1) **ML does not beat a single raw signal** — tvalpast (0.52, no ML) > ML_tval (0.43) at equal seeds=1;
  the tval EDGE is in the SIGNAL CONSTRUCTION (SE denominator), not the learner (confirms T16). CAVEAT: seeds=1 handicaps ML ~0.09
  → seeds=5 ML_tval ≈0.52 TIES raw; so fair verdict = "ML matches ONE raw signal at 5× compute + huge overfit surface" → raw wins
  on parsimony/robustness. (2) ★ **"let the portfolio sort out the signals" FAILS** — the 6 signals are 0.86–0.99 book-corr =
  literally ONE momentum bet; equal-weight composite (0.49) is WORSE than best single (0.52). A LIBRARY of momentum variants adds
  nothing; you need 1 clean signal + ORTHOGONAL sleeves, not 6 collinear ones. (3) **Better tabular ML won't help** (ML is the most
  distinct book yet the least profitable — nonlinearity buys difference, not alpha; + [[joint-representation-nn-fails]]); DL earns a
  slot ONLY via a different OBJECT (price sequence / cross-sec graph) + OBJECTIVE (direct-Sharpe/rank, Lim-Zohren), not more knobs.
  **ACTION:** momentum sleeve = ONE cheap signal (tval/tvalpast), demote per-sleeve ML; the real lever is orthogonality → next test =
  factor-momentum + autocorrelation-state timer (the one lower-corr momentum variant). GATE: seeds=1 screen; confirm ML at seeds=5.

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

### T41 — 2026-07-25 · BETA ATTRIBUTION of the champion + raw-signal control through the SAME construct layer — ★★ the sleeve has ZERO CAPM alpha; the ML's real product is BETA/VOL REDUCTION, not selection
- **Q (user "tell me the state of things"):** the champion table reports net SR 0.43–0.72 as if it were edge, while
  [[data-substrate-upgrade-2026]] found beta-neutral sleeve alphas are NEGATIVE. Which is true on honest data? And does the
  whole ML stack beat a raw momentum signal once BOTH go through the same CONSTRUCT layer?
- **Method:** fresh `CURRENT BEST/mom_layer.py` builds (seeds=1) at TOPN=1000 and full universe, honest survivorship-free
  substrate, tier=liquid, CONSTRUCT = mdv-weight + band(.10,.20), net of tiered tc+borrow. Then CAPM-regress the net monthly
  book on SPY (`data/monthly_spy.parquet`) over the live window 2011-01..2026-07 (n=188). Controls: raw 12-1, raw 6-1, and
  Blitz residual momentum (11-1 / resid-vol, 36m beta) pushed through the IDENTICAL risk_book + identical signal dates.
  Scripts: scratchpad `beta_attrib.py`, `naive_control.py`.
- **★ Result — HARNESS BUG FOUND FIRST:** `mom_layer` reports SR 0.64 / ann 7.1%, but the equity curve includes **133 warm-up
  ZERO months (2000-01..2011-01)** before the first OOS score. On the live window only: **ann 12.9% / SR 0.84**. Every SR in
  this log sourced from that harness is deflated ~25% by dead months. Numbers below are live-window.
- **★★ Result — CAPM attribution (live window, net):**
  | book | ann | vol | SR | maxDD | beta | **alpha** | t(α) |
  |---|---|---|---|---|---|---|---|
  | ML tval@6, top-1000 | 12.9% | 15.4% | 0.84 | −26.1% | 0.94 | **−0.51%** | −0.25 |
  | ML tval@6, FULL univ (rankIC .067) | 10.8% | 13.7% | 0.79 | −26.3% | 0.89 | **−2.15%** | −1.48 |
  | SPY | 14.6% | 14.1% | **1.03** | −23.9% | 1.00 | — | — |
  **The champion does not beat a beta-matched SPY** (IR −0.09 top-1000, **−0.39** full-univ). Beta-hedged the book returns
  **exactly 0.00%/yr** at 7.9% vol. R²-to-market 0.74–0.84.
- **★ Result — raw-signal control (same construct, same dates, net):** ML 0.84 SR / beta 0.94 / vol 15.4% / DD −26.1% ;
  raw 12-1 **0.64** / beta **1.40** / vol **26.7%** / DD **−49.7%** ; raw 6-1 0.64 / 1.38 / 27.5% / −38.0% ; resmom 0.72 /
  1.14 / 19.5% / −27.2%. **All four have zero-to-negative CAPM alpha** (−0.5% to −2.8%, all t insignificant).
- **★★ Conclusion — the ML stack IS worth its complexity, but it is a RISK product, not an alpha product.** vs raw 12-1 it
  delivers +0.20 SR, **−46% beta, −42% vol, −24pt DD** at LOWER turnover — exactly what tval's SE-denominator +
  reclassification + mdv-sizing are designed to do (tame the crash mode). What it does NOT do is generate selection alpha:
  **on 2011-2026 honest data no momentum book here — ML or raw — beats holding beta.** This CONFIRMS and generalizes
  [[data-substrate-upgrade-2026]] (negative beta-neutral alphas) and settles the champion table's OPEN GATE: the 0.43–0.72
  SRs are market beta, harvested at 0.9 beta with good drawdown control.
- **★ Corollary — the IC→SR inversion is now a THIRD confirmation** (T23, T29-S3, here): full-universe DOUBLES rankIC
  (.032→.067) and LOWERS net SR (0.84→0.79) and alpha (−0.5%→−2.2%). Ranking skill is real and rising; it is not tradable
  in this construction. **Stop optimizing IC.**
- **ACTIONS:** (1) the sleeve's honest role in the book is a **low-beta equity carrier**, so it must be judged against
  beta-matched SPY, not against zero — restate the champion table; (2) fix the warm-up-zeros deflation in the reported SR;
  (3) MOM alpha search on the LONG leg is at a wall — the open levers are per-name beta-neutralization (the only RISK action
  that ever helped, T39) and orthogonal premia, NOT more momentum modelling. GATE: seeds=1, one substrate, one benchmark (SPY).
- **⚠ ARTIFACT:** `CURRENT BEST/out/mom_score.parquet` is now a **seeds=1 full-universe** research build (overwrote the
  2026-07-18 seeds=5 file). Re-run `SEEDS=5` before any production use.

### T42 — 2026-07-25 · PATH SIGNATURES (level-2 / Lévy area) on top of the champion feature set — ★ small positive, NEEDS-MORE
- **Q:** `tval` is a normalised LEVEL-1 signature term of the log-price path → it is invariant to the ORDER of the moves.
  Level-2 signature coordinates are exactly the information it discards. Do they add?
- **Method:** `research/mom_research.py` STAGES=SIG. Daily-path features added to the 31-feature champion set:
  `sig_tasym{63,126}` (time-asymmetry ∫(s−s₀)dX — separates "rose then flat" from "flat then rose", same return AND
  same tval), `sig_levm` (Lévy area vs market = lead/lag), `sig_levv` (price/volume ordering), `sig_snr` (tval on 126
  daily points not 7 monthly). All rolling-sum identities → fully vectorised. seeds=1, top-1000, equal-wt `net()` reporter.
- **Result:** base(31f) IC .0454 / netSR 0.42 / DD −30.4% ; **+signature(39f) IC .0520 / 0.45 / −28.3%**, same turnover 3.5.
  Signature share of total feature importance **12.6%** (8 of 39 features) — used, but no single term in the top-12.
- **★ Conclusion — NEEDS-MORE, do not adopt.** +15% IC and +0.03 SR is inside seeds=1 noise, and T41 established three
  times that rising IC in this pipeline does NOT imply a better book — so an IC-led result is exactly the kind to distrust.
  Level-2 path information is real and orthogonal to the level-1 feature set; it is not yet shown to be tradable.

### T43 — 2026-07-25 · COST-AWARE OBJECTIVE (torch differentiable net-Sharpe) — ★ REFUTED: the objective is NOT the lever (now properly controlled)
- **Method:** STAGES=COST. Arms A champion hist-CE · B + liquidity sample weights · C torch MLP with a DIFFERENTIABLE
  NET-SHARPE loss (softmax book − per-name trading cost → −Sharpe) · D same net, rank-MSE (architecture control).
  C-vs-D isolates the OBJECTIVE from the ARCHITECTURE — the flaw in T29-S1, which used a tree proxy and wrongly
  concluded "the objective was not the leak".
- **Result (IC6m / net SR / ann / maxDD / turn):** A champion hist-CE .0454/**0.42**/4.1%/−30.4%/3.5 ;
  B + liquidity sample weights .0403/**0.42**/4.2%/−29.1%/3.5 ; C torch NET-SHARPE .0345/**0.38**/4.5%/−33.3%/3.6 ;
  D torch rank-MSE (control) .0486/**0.38**/3.9%/−27.7%/3.6.
- **★ Conclusion — the OBJECTIVE is not the leak.** C vs D holds the network and features fixed and varies ONLY the
  loss: **identical net SR (0.38 vs 0.38)**. A differentiable net-of-cost Sharpe objective buys NOTHING over plain
  rank-MSE. Liquidity sample weighting (B) also does nothing (0.42 → 0.42). Torch (0.38) < XGBoost (0.42), so the NN
  does not beat trees either. This UPHOLDS T29-Stage-1's conclusion, which I had criticised as a mere tree proxy —
  the critique of the DESIGN was fair but the ANSWER was already right.
- **★ Where cost DOES belong:** the PORTFOLIO objective, not the model loss. Gârleanu-Pedersen is a closed-form
  convex solution and `execution_layer.py` already implements it — T46 shows it flips the Han-DM book's net alpha
  from −11.1% to +3.9% and cuts turnover 7.1→2.0. Putting cost in the LOSS is the wrong layer; putting it in the
  TRADE RATE is the right one. Also worth testing at the RANKING step (rank on μ̂ − round-trip cost) — see
  `research/dm_criteria.py`.

### T44 — 2026-07-25 · BETA MANAGEMENT overlays on the champion long book — ★ timing REFUTED; vol-managing weakly positive
- **Q:** T41 showed the sleeve is a 0.9-beta carrier with 0.00% alpha. Can beta be MANAGED into alpha?
- **Method:** STAGES=BETA. Overlays on the champion long book's net returns, all dials PIT-lagged. A base · B static
  rolling-36m hedge · C STATE-timed hedge (expanding-z credit+VIX stress PC) · D vol-managed (12% target, Barroso-
  Santa-Clara / Moreira-Muir) · E Goulding-Harvey-Mazzoleni four-state exposure (FIXED prior exposures, not fitted)
  · F B+D. Judged on ALPHA and IR vs SPY, not SR.
- **Result (ann / SR / maxDD / beta / alpha / t):** A base 13.14%/0.85/−26.1%/0.94/**−0.36%**/−0.17 ; B static hedge
  −0.89%/−0.11/−42.1%/−0.04/−0.25%/−0.12 ; C STATE-timed 10.32%/0.73/−26.1%/0.70/+0.30%/**0.11** ; **D vol-managed
  12.65%/0.91/−22.8%/0.80/+1.25%/0.57** ; E four-state 10.35%/0.82/−28.2%/0.68/+0.62%/0.29 ; F B+D 0.15%/0.02/−33.9%.
- **★ Conclusion — beta management cannot create alpha.** Arm B independently reproduces T41: strip the beta properly and
  the book returns −0.89%/yr. STATE-timed hedging does NOTHING (t=0.11) — the "point STATE at beta instead of alpha"
  reframe is REFUTED. Four-state also fails (t=0.29). Only vol-managing helps and only modestly (+0.06 SR, +3pt DD,
  t=0.57 = not significant) — take it as free construction hygiene, not a finding.

### T45 — 2026-07-25 · ★★ HAN RECONCILIATION — our "dollar-neutral" L/S carries beta ≈ −0.85; the "no short premium" conclusion is an ARTIFACT
- **Q (user "how can this be all beta?!"):** Han reports a VW L/S alpha of 2.4%/mo (t=6.63) and Quantitativo replicate
  SR 1.78 at beta −0.111. We measured beta 0.9 / zero alpha. Reconcile.
- **★ FINDING 1 — different portfolio object.** Han's headline is a DOLLAR-NEUTRAL L/S (beta≈0 by construction, so its
  whole return IS alpha). We measured a LONG-ONLY mdv-weighted book. Han's OWN cost-realistic long-only number is
  **SR 0.34** (§1.3, DeMiguel costs) — below the market of his era. **Han never claims long-only alpha.** Our T41
  long-only result is therefore CONSISTENT with Han, not in conflict.
- **★★ FINDING 2 — our L/S was never beta-neutral.** Measured (net, 2011-2026): liquid/mdv **L/S beta −0.89, alpha
  +9.42% (t=1.25)**; equal L/S beta −0.83, alpha −5.09%. Dollar-neutral ≠ beta-neutral: the short leg is loaded with
  high-beta losers (Daniel-Moskowitz) so equal DOLLAR weights leave ≈ −0.85 net beta. In a 15-year bull market that
  alone produces the loss. **T35/T37/T39/T40 all measured raw return on a −0.85-beta book and read the market exposure
  as missing alpha. The four "independent confirmations" of "no short premium" share one defect.** RISK's beta_neut
  only reaches −0.39, so even the "beta-neutral" arm is not neutral.
- **★ FINDING 3 — period.** mdv long-only alpha +0.88% (2011-2017, Han-overlap) → **−3.72% (2017-2026)**; equal
  long-only −1.31% → **−8.73% (t=−4.05)**. Post-publication decay, sharply after Han's sample ends.
- **FAILED AXIS:** the `relaxed` vs `liquid` tier rows are IDENTICAL — the score parquet was already built under the
  liquid filter, so `risk_book` cannot add unscored names. Testing Han's "no liquidity filter" needs a POOL rebuild.

### T46 — 2026-07-25 · ★★ FAITHFUL HAN REPLICATION (DM.py rebuilt) + DynTrad — ★ the alpha is REAL, and it is entirely in the ILLIQUID TAIL
- **Method:** `DM.py` rewritten as Han's exact spec (old sleeve → `research/archive/DM_legacy_multihorizon.py`):
  20 features (5 nMOM + 5 **M_MOM cross-sectional means = the macro-state features WE NEVER HAD** + 10 size dummies);
  DNN 5×64 ReLU softmax-10, CE, early stopping ONLY; EXPANDING window, ANNUAL refit, last-10y validation;
  Return-reclassification with μ̂_k = 20-year EMPIRICAL class means; decile books, EW and VW; NO tier filter.
  Pool 425 months, **3,560 names/month** (Han: 3,837). seeds=1, test 2011+.
- **★ Result GROSS (EW):** H 35.2%/SR 1.12/beta 1.62/α +12.05% ; L −9.6%/beta −0.98/**α +4.37%** ; **HL 25.6%/SR 1.32/
  maxDD −14.3%/beta 0.64/α +16.43% (t=3.62)** ; HL borrowable **α +25.11% (t=5.07)** ; **HL BETA-NEUTRAL SR 1.72,
  α +26.27% (t=5.63)**. (SIZE=0 variant: HL α +19.92% t=4.67, beta-neut +28.49% t=6.00.) Han: EW HL >40%/SR 2.0-2.49.
- **★★ Result — THE KILLER:** **`EW HL LIQUID universe` GROSS alpha −2.59% (t=−0.79)** and `EW H LIQUID` gross α −15.64%
  (t=−3.80). **The entire alpha lives in the illiquid tail and is gone in tradable names BEFORE any cost.** NET, every
  arm is negative (HL full univ −54.92%, turnover 7.2).
- **Result — DynTrad (`execution_layer.py`, GP-2013, previously ZERO callers) on the Han HL book:** raw net α −11.12%
  (t=−2.34), turn 7.1 → **λ=2: net ann +7.5%, SR 0.46, maxDD −33.4%, α +3.93% (t=0.93), turn 2.0.** GP FLIPS THE SIGN
  of net alpha and cuts DD by 41pt. But λ=10+ over-slows and destroys it (α −1.02% → −10.81%), and
  `calibrate_lambda` self-selects **λ=16.6 → α −5.15%, i.e. the "correct" calibration picks a much WORSE point than
  the paper default λ=2.** CALIBRATION BUG — do not trust `DynTrad.from_book` as-is.
- **★★ Conclusion — the diagnosis changes from "momentum has no alpha" to "momentum's alpha is not tradable at this
  turnover/capacity".** Han's method DOES produce strong beta-independent alpha on our data (+26%, t=5.63) and the
  SHORT leg contributes positively once beta is handled — but ~85% of it is paid away between signal and fill, and the
  liquid-universe slice has none of it gross. Size dummies (SIZE=1 mcap / SIZE=2 mdv-proxy) do NOT reproduce Han's
  large-cap shift (§4.4.1) — SIZE=2 is slightly WORSE. Levers, in order: (1) wire DynTrad in properly + fix its
  calibration, (2) a MULTI-FACTOR risk model as optimizer CONSTRAINTS (RISK.py has none — only a single-factor beta
  projection; `research/cov_model.py` has 0 callers), (3) accept momentum's small tradable slice and add ORTHOGONAL
  sleeves. GATE: seeds=1, one substrate.
- **DATA BUG FOUND:** `hub.mcap` is EDGAR-derived and EMPTY before ~2009 (1-2 names/month pre-2007; ~35% coverage even
  in 2024). Any sleeve touching mcap is implicitly a 2010+ strategy. It silently truncated the first Han run to a model
  trained on 25 months. VW books are unreliable for the same reason (non-filers get zero weight).
- **ENGINE BUG:** `BACKTEST._metrics` raises `TypeError: float() argument ... not 'complex'` when equity goes ≤0
  (negative equity → complex power in the Sortino downside term). Guard it.

### T47 — 2026-07-25 · COST MODEL REBUILT (measured IBKR + Corwin-Schultz) — ★★ the old tiers were 7× too punitive on the median name
- **Q (user "the costs need to be accurate — Interactive Brokers"):** is `BACKTEST.DEFAULT_COST_TIERS` right?
- **★ Finding:** it charged the Han-DM book a mean **93 bp ONE-WAY** (liquid 56, full univ 132) vs Frazzini-Israel-
  Moskowitz's MEASURED institutional median of **6.2 bp**. Structurally wrong in both directions: IBKR commission is
  charged PER SHARE so cost in bp scales with **1/price**, not with liquidity (a $200 stock = 0.18 bp, a $2 stock =
  17.5 bp at identical dollar volume — the tiers charge both 5 bp).
- **Built `research/ibkr_costs.py`:** commission (IBKR Pro $0.0035/share, min $0.35, cap 1% of value) + **MEASURED**
  half-spread (Corwin-Schultz 2012 from our own daily HIGH/LOW) + SEC §31 + optional √-impact. Two corrections that
  matter: non-trading days masked (when a stock does not trade H==L and CS collapses toward ZERO, reporting the least
  tradable names as the CHEAPEST) and a TICK FLOOR (half-spread ≥ $0.005/price — a physical bound).
- **★ VALIDATION:** mega-cap one-way = **6.9→7.6 bp** vs FIM's measured **6.2 bp**. Independent, not fitted.
  New medians: all names **25.3 bp** (was 150) · mdv>$5M 17.4 (was 25) · mdv>$100M 11.8 (was 10) · mdv>$1B 7.6 (was 5).
- **★★ IMPACT ON CONCLUSIONS:** re-running the Han-DM book with measured costs turned `sharpe L/S` from
  alpha **−1.37% (t=−0.54)** into **+9.68% (t=3.78)**, SR 0.88, beta −0.08, maxDD −13.6%. **The old tiers were
  manufacturing the failure on every high-turnover/full-universe book in this log.** LEAST-ACCURATE PIECE REMAINING =
  BORROW (no historical IBKR SLB feed; tiers recalibrated to published GC but still assumed).

### T48 — 2026-07-25 · CAPACITY & SIZE ATTRIBUTION — ★★ the alpha is real, monotone in size, and caps out around $10M AUM
- **Method:** `research/capacity_aum.py`, `research/size_attribution.py`. Square-root impact k·√(|Δw|·AUM/ADV), k=0.10.
- **★ CAPACITY (sharpe L/S, full univ):** no-impact +9.68% (t=3.78) · **$1M +5.10% (t=2.03)** · $10M +1.26% (t=0.51) ·
  $50M −3.79% · $1B −24.12%. **Honest capacity ≈ $5-10M; significant only below ~$1-2M.**
- **★ SIZE ATTRIBUTION (same signal, per size quintile, self-financed, net):** Q1 smallest gross α +66.11% (t=6.44) /
  net +14.54% (t=1.63) · **Q3 middle gross +24.75% / NET +9.67% (t=2.72), SR 0.69** · Q5 largest gross +5.16% (t=2.14)
  / net +2.05% (t=0.85). Alpha is MONOTONE DECREASING in size but does NOT vanish at the top — and the best NET bucket
  is the **MIDDLE**, not the smallest (Q1 has 2.7× the gross alpha but loses it to cost). We had been testing
  "full universe vs liquid" as a binary and missing the sweet spot.
- **Book size profile:** the Sharpe criterion DOES shift the long book larger (1.40× universe median) — Han's §4.4.1
  direction — but the SHORT book shifts SMALL (0.32×), the opposite of Han. Half-reproduced.

### T49 — 2026-07-25 · SIGNAL COMBINATION — ERC/equal/netting all FAIL; SHRUNK MVO WINS — ★★ and position-netting is REFUTED
- **Q (user):** run ERC on the raw signals; and "isn't this how RenTec trades, millions of signals combined"?
- **Corrected engine throughout:** Q3, hold=6, banded, MEASURED IBKR costs. Raw signals beat the ML decisively:
  hi52 net α **+37.10%** (t=4.84) · mom11 +29.64% (SR **1.03**) · composite +27.32% · tvalpast +21.06% · resmom
  +19.14% · **ML_sharpe +9.67%** · fip +2.30% (t=0.68). **The Han DNN is 3-4× worse than arithmetic** — T26 confirmed
  and strengthened. Book correlations 0.38-0.85 (LOWER than T26's 0.86-0.99; the engine changes the redundancy picture).
- **★ ERC IS THE WORST COMBINER:** best-single SR 1.03 > score-composite 0.76 > equal-weight 0.79 > inv-vol 0.70 >
  **ERC 0.68**. WHY, diagnosably: ERC equalises RISK contribution → up-weights LOW-VOL books → piles into `fip`, the
  one with no alpha. Risk parity assumes EQUAL SHARPE; ours run 1.03 to −0.17. `ERC.py`'s own docstring says
  "for a FEW similar-Sharpe sleeves" — the tool was outside its stated domain.
- **★★ POSITION-LEVEL NETTING REFUTED (corrects my own earlier claim):** netting cut turnover 1.8→1.4 as predicted but
  LOST 1.9-2.5pt of alpha — more than the cost saving. Netting cancels exactly the positions where signals DISAGREE,
  and disagreement carries information. Internalisation is not free for correlated sleeves.
- **★★ SHRUNK MVO WINS:** `w ∝ (Σ+λI)⁻¹μ` → **SR 1.08, maxDD −23.6%** (vs mom11 1.03/−70.5%, hi52 0.84/−80.5%).
  Flat in λ (SR 1.00-1.08 over five orders of magnitude). **Kozak-Nagel-Santosh vindicated on our data: sparse 1.03 <
  naive dense 0.79 < dense+shrinkage 1.08.** The −47pt drawdown improvement is the headline. My "one signal per
  premium, don't combine" was WRONG.
- **★ THE CEILING — `N_eff = 1.80` of 5** (eigenvalues [3.64,.70,.35,.22,.10]; one PC = 73% of variance). Grinold
  IR≈IC·√breadth ⇒ five 0.85-correlated signals buy √1.80 = **1.34×**, not √5 = 2.24×. **No combiner can exceed this.**
  Adding a 6th momentum variant moves N_eff by ~0.05. Breadth requires new DATA, not new arithmetic on one price path.

### T50 — 2026-07-25 · IPCA (Kelly-Pruitt-Su) — ⚠️ UNRESOLVED (implementation flaw), but the DIAGNOSIS is solid
- **Method:** `research/ipca_model.py`, restricted IPCA r=β'f with β=z'Γ, ALS, annual refit, expanding window.
- **Result:** every K gives SIGNIFICANTLY NEGATIVE alpha (K=1 −22.59% t=−3.22 … K=3 −10.81% t=−2.62), and a Γ-ridge
  sweep (0→100) does NOT rescue it. **REASON IT IS UNRESOLVED:** I shrink Γ and then orthonormalise to Γ'Γ=I, which
  rescales the penalty away — the shrinkage is ~a no-op. Do NOT read this as a verdict on IPCA.
- **Two REAL bugs found and fixed (neither changed the prediction):** (a) `np.linalg.qr` returns Q unique only up to
  sign(diag(R)) so Γ's columns could flip while F did not; (b) the Γ normal equation had BOTH Kronecker products
  reversed and a row-major reshape — the FOC is ΣZ'ZΓff'=ΣZ'rf' so vec(AXB)=(Bᵀ⊗A)vec(X) needs (ff'⊗Z'Z),
  RHS kron(f,Z'r), reshape order="F". Neither moved the forecast because **Γf̄ is invariant to the parameterisation**
  ((Γ,f)→(ΓA,A⁻¹f) leaves Γf fixed and f is re-fit by least squares) — that invariance is what redirected the search.
- **★★ THE DIAGNOSIS (solid, and it came from plain OLS/Fama-MacBeth, no IPCA):** every signal is positively
  predictive UNIVARIATELY and SIGN-FLIPS when estimated JOINTLY — hi52 univariate rank-IC **+0.0691**, pooled-OLS
  coefficient **−0.01103**, Fama-MacBeth **−0.00836**. With N_eff=1.80 the design matrix is near-singular and the
  joint coefficients are noise on the near-null directions. **This is a standing hazard for ANY joint model over the
  signal library, trees included, and may explain T26/T29-S3's signature (highest IC, lowest SR).** It is also exactly
  the failure KNS's shrinkage exists to prevent — which is why the shrunk MVO (T49) works and unshrunk IPCA does not.
- **ACTION:** built `research/signal_combiner.py` — KNS eigenvalue-proportional shrinkage + James-Stein on μ + an
  ADMISSION TEST for new signals (ΔN_eff · book-corr to composite · spanning alpha BOTH ways). Judge on the BOOK, never
  the score (T31 read 0.15 on scores, T33 read 0.74 on books for the same idea).

### T51 — 2026-07-25 · ADMISSION TEST + FLOWS DATA — ★★ BEST BOOK OF THE SESSION (SR 1.25), and it corrects two of my own claims
- **Built `research/signal_combiner.py`:** reusable library — `add(name, scores)` / `combined()` / **`admit(name, scores)`**.
  The admission test judges a CANDIDATE on INDEPENDENCE, not standalone Sharpe: ΔN_eff · book-corr to the composite ·
  spanning alpha BOTH ways. All statistics on NET BOOK returns, never scores (T31 read 0.15 on scores, T33 read 0.74 on
  books for the same idea). KNS eigenvalue-proportional shrinkage + James-Stein on μ.
- **★ CORRECTION 1 to T49 — shrinkage's win was mostly EXCLUDING A BAD INPUT.** With `fip` in the library equal-weight
  = 0.79 and ridge = 1.08. Drop `fip` and **equal-weight = 1.07**, ridge = 0.93. The shrinkage was earning its keep by
  down-weighting the one worthless signal; simply not including it does the same job. `N_eff` of the 4 good signals is
  **1.43** (LOWER than 1.80 with fip — fip was the most distinct one).
  ⚠️ MY BUG: `signal_combiner`'s isotropic-ridge arm uses `kappa*I` unscaled by the covariance magnitude, so at monthly
  return scale the penalty swamps Σ and all κ collapse to one answer (0.93/0.93/0.92). Not comparable to T49's ridge.
- **★ ADMISSION RESULTS (core = hi52/mom11/tvalpast/resmom, N_eff 1.43):**
  | candidate | SR | corr→comp | ΔN_eff | α&#124;comp (t) |
  |---|---|---|---|---|
  | fip (momentum) | −0.08 | −0.26 | **+0.37** | +2.26% (0.53) |
  | svol_ratio (flows) | −0.07 | +0.12 | **+0.42** | −2.15% (−0.62) |
  | ftd_days (flows) | −0.29 | −0.19 | **+0.61** | −4.70% (−0.41) |
  | svol_abn (flows) | −0.20 | +0.21 | **+0.56** | −3.29% (−1.32) |
  **The flows signals DO raise effective breadth (ΔN_eff up to +0.61 — far more than any momentum variant) but NONE has
  positive spanning alpha.** So the "different data source ⇒ new premium" thesis is only HALF supported: the data is
  genuinely independent, the ALPHA is not there standalone.
- **★★ CORRECTION 2 — but they still pay, through COVARIANCE.** Adding `svol_ratio` to the library:
  core-4 equal SR 1.07 / maxDD −67.5% → **core-4 + svol_ratio with KNS κ=0.5: SR 1.25, vol 16.8%, maxDD −40.3%,
  alpha +18.39% (t=3.70)** — the **best risk-adjusted book of the session**. A signal with NO standalone alpha improved
  the portfolio by 0.18 SR and 27pt of drawdown purely as a diversifier, and only the SHRUNK combiner captured it
  (equal-weight with it = 1.01, i.e. WORSE than without).
- **★ LESSON — the admission test's own threshold is wrong.** It required spanning-alpha t>2.0, which would REJECT
  `svol_ratio` — the input that produced the best book. A signal can earn its place through COVARIANCE alone. Fix:
  admit on the marginal change in COMBINED portfolio Sharpe, not on the candidate's spanning alpha.
- **★ SYNTHESIS OF THE WHOLE SESSION.** Estimator changes (features T26/T42, objectives T29-S1/T43, architectures T46,
  combiners T49, IPCA T50) were worth ≤0.05 SR and mostly negative. DATA and CONSTRUCTION changes were worth 0.3-1.0 SR:
  cost-model accuracy (T47) −1.37%→+9.68% alpha · holding period (T49) −12.45%→+1.46% · size bucket (T48) +9.67% t=2.72 ·
  orthogonal data + shrinkage (T51) SR 1.07→1.25. **The bottleneck was never the model.**

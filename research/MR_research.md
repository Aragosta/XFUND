# MR_research — working thesis & test log (MEAN-REVERSION sleeve)

**Living document.** Working thesis at the top; every test appended to the LOG with its EXACT method + conclusion.
Same discipline as [RESEARCH_PROTOCOL](RESEARCH_PROTOCOL.md) (UNDERSTAND → DEFINE → MEASURE → ANALYZE). Sibling
logs: `MOM_research.md`, `VALUE_research.md`. Rebuilt from scratch 2026-07-18 (old reversion scripts removed;
findings preserved in memory [[reversion-qp-sleeve]] [[reversion-catalog-synthesis]]).

---

## ⚠️ MANDATORY READING — read BEFORE proposing/testing/claiming anything on this sleeve (Phase R)
The MR sleeve's backbone comes from the Quantitativo QP series (read all three; they build the same strategy up):
1. **"A mean-reversion strategy from first principles"** (quantitativo.com/p/a-mean-reversion-strategy-from-first) — the
   **QP indicator** (rarity of a 3-day move vs the stock's own 5-yr distribution, 0–100, low=rare) + **200-day-SMA trend
   gate** + **event exit** (close > yesterday's high → exit next open). Rule-only: SR 1.10 / DD 26% (S&P 500, 22y). ★MANDATORY
2. **"Machine learning and the probability"** (quantitativo.com/p/machine-learning-and-the-probability) — **XGBoost binary
   classifier**, target = **P(bounce within 5 days)** (1 if +ret), 16 features (ROC multi-horizon, RSI, QPI-windows, IBS,
   norm-ATR, dist-to-200SMA, turnover cross-sec+ts, **Hurst**), cross-sec standardised; used as a **FILTER** (enter iff
   QP<15 AND P>60%). Lifts SR 0.69→1.33 but DD stays ~50%. ★MANDATORY
3. **"Long and short mean reversion machine"** (quantitativo.com/p/long-and-short-mean-reversion-machine) — the L/S build:
   short = mirror (rare POP in downtrend), **short sized ~15–20% of long** (short is riskier), **VIX regime filter** (bear
   when VIX>15d-SMA×1.15 = 90th pct → cut long exposure 1.1x→0.1x), ≤20+20 names, pos ≤5% of 3m ADV, Russell-3000 PIT.
   SR 1.55 / DD 19% (2014–24, WITH 2020). ★MANDATORY

**HONEST CAVEAT (do not skip):** these are Quantitativo's numbers at **2 bp, their engine, WITH COVID-2020** (removing 2020
drops the flagship to ~26% ann). Our own honest-engine review ([[reversion-catalog-synthesis]]) found reversion nets only
**~0.2–0.3 after look-ahead+cost stripping in the liquid universe.** Treat 1.1–1.55 as a CEILING; the deployable question is
the honest daily engine (lag=2, per-side bps). The STRUCTURE is the keeper; the SR is not.

---

## THE FRAMEWORK — reversion is a CLASSIFICATION (mixture-separation) problem, not a rule
*(This is the NEW direction from the QP-ML articles. Prove it before coding — Phase 1 UNDERSTAND.)*

**The idea (Quantitativo QP-ML).** The oversold rule (QP<15) fires on a MIXTURE: some oversold stocks are
**liquidity-driven overreactions that BOUNCE**, others are **information-driven declines that KEEP FALLING** ("falling
knives"). Buying all of them earns the *blended* expected return — near zero, statistically insignificant. **Train a
classifier to predict P(bounce ≤ 5d) and trade only the high-probability subset** → you've *separated the mixture* and
kept the reverting component. This is the SAME conditional-expectation logic as DM in the MOM sleeve (bimodality →
classify → condition on the good component), applied to the reversal mixture instead of the momentum mixture.

**Mathematical justification (mixture model → conditioning raises the mean).** Let the 5-day forward return of an
oversold event be a two-component mixture indexed by a latent state S ∈ {Liq, Info}:

  r | oversold  ~  π·𝒩(μ_L, σ_L²)  +  (1−π)·𝒩(μ_I, σ_I²),   μ_L > 0 (bounce),  μ_I < 0 (knife).

The **unconditional** edge is `E[r | oversold] = π·μ_L + (1−π)·μ_I` — the two components OFFSET, so it's small and noisy
(matches the article: rule +0.7%/trade, p=0.06, insignificant). Features x shift the **posterior** `p(x)=P(S=Liq | x)`.
A classifier estimates p̂(x); selecting `{x : p̂(x) > c}` gives

  E[r | oversold, p̂(x)>c]  =  E[π(x)·μ_L + (1−π(x))·μ_I  |  π(x)>c]  ↑  as c ↑,   →  μ_L in the limit.

i.e. **conditioning on the classifier monotonically raises the conditional mean toward the bounce component** (and cuts
variance by dropping the offsetting knives). That is exactly the empirical result: +0.7%→+1.6%/trade, p→<0.05. The
classifier is doing *mixture separation / posterior thresholding* — Bayes, not curve-fit. Same object as Han's DM
`RET = Σ pₖμₖ` (reclassification = conditional expectation over a mixture); here the mixture is {bounce, knife}.

**What the features are proxying (why these features).** They separate liquidity-overreaction from information:
- **Hurst exponent** (Hurst 1951; Lo 1991) — *directly* measures mean-reversion propensity (H<0.5 revert, >0.5 trend). The
  natural S-separator.
- **QPI multi-window / RSI / ROC** — rarity & speed of the move (bigger/rarer/faster drops on no trend-break = overreaction).
- **turnover (ts + cross-sec), normalized ATR, IBS** — liquidity/volatility state; **dist-to-200SMA** — is the drop a blip in
  an uptrend (liquidity) or a break of trend (information). Note: 200-SMA is a **FEATURE here, not a hard gate**.

**Academic grounding (cross-reference — this is not a blog trick).**
- *Reversal exists:* **Jegadeesh (1990); Lehmann (1990); Lo & MacKinlay (1990)** — short-horizon return reversal / contrarian profits.
- *Reversal = liquidity-provision premium (the "Liq" component):* **Nagel (2012, RFS), "Evaporating Liquidity"** — short-term
  reversal returns are compensation for providing liquidity; they SPIKE with VIX/illiquidity. This is *why* a regime/stress
  state matters and *why* the premium is real, not a risk premium.
- *The falling-knife distinction is real & documented:* **Da, Liu & Schaumburg (2014, RFS), "A Closer Look at the Short-Term
  Return Reversal"** — decompose reversal; the profits come from the **non-fundamental (liquidity) component**;
  information/fundamental moves do NOT revert. This is the academic backbone of "classify bounce vs knife."
- *Stat-arb construction:* **Avellaneda & Lee (2010)**. *ML for conditional expected returns:* **Gu, Kelly & Xiu (2020)** — the
  general result that ML-estimated *conditional* expectations beat unconditional signals out of sample.

**Regime gate = STATE, not VIX.** Nagel (2012) says the premium is liquidity-provision — largest but riskiest in stress.
So the off-switch is the **STATE layer's stress** (`out/state_dial.parquet`: `surprise` = Mahalanobis turbulence, `gross`
= vol-target dial), reused across sleeves — NOT a bespoke VIX rule. To test: does gating the book by STATE stress cut the
−51% DD without killing the edge (T5)?

**Predicted result before running (pre-registration).** Unconditional QP ≈ marginal (T1 confirmed 0.44 net). ML-conditioned
should lift the per-trade mean and Sharpe (mixture separation is real), BUT (a) honest costs at 190× turnover will eat much
of it, and (b) the DD needs the STATE gate. Net-deployable prior: modest, **judged as a ~0-corr diversifier to MOM+DM**, not
a standalone star. If ML-conditioned net-SR ≤ unconditional after honest costs, the classifier isn't separating the mixture
on OUR universe/engine → reject.

### THE TARGET — the binary bounce is TOO COARSE; denoise it, encode SPEED, and Stop-Regressing applies
The article's label (`1[r_{5d} > 0]`) is the **K=2 special case** — the MR analog of DM's 2-bucket. It throws away
**magnitude** (a +0.3% bounce = a +8% bounce) and **speed** (revert in 1 day vs drift back over 5). Two upgrades, both
already proven in the MOM sleeve and now academically motivated for MR:

**(a) Reversion SPEED belongs in the target (academically, not just intuitively).** **Dai, Medhat, Novy-Marx & Rizova
(2023, FAJ), "Reversals and the Returns to Liquidity Provision"** show reversal SPEED is systematic: **high volatility →
faster, initially larger reversals** (higher inventory risk → liquidity providers demand a quicker snap-back); **low
turnover → slower, longer-lived reversals** (longer inventory duration). So faster reversion is both *more desirable*
(less holding risk, less capital-time) *and predictable from vol/turnover*. → Target should reward **fast** mean-touch,
and **realized-vol + turnover are first-class features** (they set the expected speed). Concretely, a **speed-weighted
reversion target**: e.g. `bounce_return / days_to_mean_touch` (or discount the 5-day return by days-to-first-`close>prev-high`),
so a 1-day snap scores far above a slow 5-day drift-back. This is the reversion analog of MOM's tval = "drift ÷ noise":
here it's "**reversion ÷ time**".

**(b) Stop-Regressing / HL-Gauss applies here too — YES.** The 5-day reversion return is **noisy and fat-tailed**
(short-horizon single-name returns are the fattest-tailed target we have), so MSE-regressing it would explode on the
tails exactly as in MOM. The fix is identical: **predict a HISTOGRAM over the (speed-weighted) reversion outcome with
cross-entropy (HL-Gauss), then read `RET = Σ pₖμₖ` back out and rank** — bounded gradient, optimisation-stable
(Farebrother 2024; Imani-White 2018). The binary classifier is the coarse K=2 collapse of this.

**★ THE UNIFICATION (cross-sleeve).** All three sleeves are then **ONE architecture**: *histogram over a sleeve-specific
target → cross-entropy (HL-Gauss) → `RET = Σ pₖμₖ` reclassification → rank*. The **only** sleeve-specific choice is the
TARGET:
| sleeve | target (what the histogram is over) |
|---|---|
| DM | forward return |
| MOM | tval = trend drift ÷ path-noise |
| **MR** | **speed-weighted reversion = bounce ÷ time (denoised, not binary)** |
The MR "innovation" (ML bounce classifier) is just DM's histogram→RET with a reversion target — and denoising it
(speed-weight) + distributional-ising it (HL-Gauss) is the same move that lifted MOM. Prove/measure before adopting:
does the speed-weighted histogram target beat the coarse binary on net-SR after honest costs? (planned test.)

### RECENT MEAN-REVERSION LITERATURE (last ~10y — the ones that shape this design)
1. **Dai, Medhat, Novy-Marx & Rizova (2023, FAJ) — "Reversals and the Returns to Liquidity Provision"** (NBER w30917).
   Reversal = liquidity-provision return; **vol → fast reversal, turnover → slow reversal.** The speed-target backbone. ★
2. **Da, Liu & Schaumburg (2014, RFS) — "A Closer Look at the Short-Term Return Reversal."** Non-fundamental (liquidity)
   component reverts; fundamental (information) does not = the bounce-vs-knife split the classifier separates. ★
3. **Nagel (2012, RFS) — "Evaporating Liquidity."** Reversal = paid for providing liquidity; spikes in stress → STATE gate. ★
4. **Cheng, Hameed, Subrahmanyam & Titman (2017, JF) — "Short-Term Reversals: Past Returns & Institutional Exposure."**
5. **Ignashkina, Rinne & Suominen (2022, JBF) — "Short-term reversals, returns to liquidity provision & costs of immediacy."**
6. **Della Corte & Kosowski — "Market Closure and Short-Term Reversal."**
7. **"Is short-term reversal driven by liquidity provision in emerging markets? Evidence from China" (2022).**
8. **Gu, Kelly & Xiu (2020, RFS) — "Empirical Asset Pricing via ML"** — reversal is a top ML feature; conditional > unconditional.
9. **Li et al. (2024) — "Real-time Machine Learning in the Cross-Section of Stock Returns."**
10. **Avellaneda & Lee (2010) — "Statistical Arbitrage in the U.S. Equities Market"** (construction reference).
Common thread of the modern literature: **reversal is liquidity provision; its speed/strength is governed by
vol & turnover; only the non-fundamental component reverts** — all three point to *conditioning + a speed-aware target*, exactly our design.

---

## STARTING POINT — ML v1 spec (target · features · model) — the concrete thing to implement & test
*Same architecture as DM/MOM (histogram → RET reclassification); only the TARGET is reversion-specific.*

**TARGET — speed-weighted, distributional (the two upgrades over the article's binary).**
- `y_t = Σ_{d=1}^{H} λ^{d-1} · r_{t+d}`  — **time-decayed H-day forward return** (H=5, λ≈0.7). A day-1 bounce counts far
  more than a day-5 drift-back → encodes **reversion SPEED** (Novy-Marx: fast reversal is the desirable/predictable kind).
  Continuous (denoised vs `1[r₅ₐ>0]`). Long a NAME iff its predicted `y` is high (fast bouncer).
- Rank `y` cross-sectionally → **10-bucket histogram → `multi:softprob` (HL-Gauss)** → read `RET = Σ pₖ·centerₖ`
  (Gaussian centers). Binary bounce = the K=2 collapse of this; the histogram is strictly richer + tail-stable (Stop-Regressing).

**FEATURES — reversion set (cross-sec Gaussian-ranked, daily, PIT frozen at t−1).**
- *Rarity/trigger:* QP at k∈{3,5,10} (oversold event + its rarity).
- *Speed predictors (Novy-Marx):* **per-name realized vol {21,63d}** (IDIOSYNCRATIC, cross-sectional — NOT the STATE
  layer, which is low-dim systematic/portfolio vol; per-name vol should live as a global `hub.rvol` like `hub.elig`,
  shared by all sleeves); turnover = dollar-vol as **ts-relative** (vs own trailing) **and cross-sec rank**. These set
  the expected reversion speed and are load-bearing. (Systematic/market vol = STATE `sig_port`/`surprise` → the GATE, not a feature.)
- *Momentum/trend context:* ROC {5,21,63,126,252}; **dist-to-200SMA (a FEATURE, not a gate)**; RSI {2,14}.
- *Mean-reversion propensity:* **Hurst exponent** (rolling) — the direct Liq-vs-Info separator.
- *(If OHLC available:* IBS, normalized-ATR; else close-based vol proxies.)

**MODEL.** XGBoost `multi:softprob`, 10-class over the rank-bucketed speed-target → `RET=Σpₖμₖ`. **Walk-forward, 15-yr
rolling window, annual retrain** (per the articles), cross-sectional. Seeds-ensembled. Identical estimator to DM/MOM.

**CONSTRUCTION.** Rank on RET → long top-decile (fast bouncers) + short bottom-decile **BORROWABLE only**
(`hub.elig('shortable')`), dollar-neutral, short sized ~15–50% of long, breadth-weighted (~20+20 names), ~5-day hold /
event exit. **Honest daily engine** (signal[d]→trade close[d+1]→earn d+2 = lag=2; per-side bps; borrow). Any lag-0 = mirage.

**REGIME GATE = STATE (not VIX).** Scale the book by STATE stress (`CURRENT BEST/out/state_dial.parquet`: `surprise`
turbulence / `gross` dial) — cut exposure in turbulence (Nagel: the liquidity-provision premium is riskiest in stress).

**PRE-REGISTERED FIRST TESTS (Phase 2 DEFINE — measure before believing):**
- **T2** — ML histogram→RET (speed target) vs the rule baseline (T1 = 0.44) AND vs the article's binary classifier, net of honest cost. Bar: net-SR Δ ≥ +0.10.
- **T3** — does the **speed-weight** (λ<1) beat the plain H-day return target (λ=1)? λ ∈ {1, 0.85, 0.7, 0.5} sweep.
- **T4** — STATE-stress gate vs raw: does it cut the −51% DD without killing SR?
- **T5** — vol/turnover feature ablation: are the Novy-Marx speed features actually load-bearing (drop them → does edge fall)?
- **T6** — corr to MOM+DM + ERC book value (the real deployment test — it's a ~0-corr diversifier or it's nothing).

---

## THE WORKING THESIS (UNDERSTAND — as of 2026-07-18, starting fresh)

**1. What the premium IS.** Short-horizon (days to ~2 weeks) *overreaction correction* / liquidity-provision:
a name that gaps down without news tends to bounce; a name that pops tends to fade. It is the **opposite sign of
momentum at a much shorter horizon** — momentum harvests at 6–12 months, reversion at ~1 week. The economic source
is **liquidity provision** (you get paid to absorb forced/impatient order flow) + behavioral overreaction, NOT a
risk premium. That origin dictates everything below: it's high-turnover, capacity-limited, and cost-sensitive.

**2. Why it earns its place — ORTHOGONALITY, not standalone SR.** Reversion's value to the book is that it is
**~0 correlation to MOM/DM** (different horizon, opposite sign) — a genuine diversifier. So the bar is NOT "beat
MOM's SR"; it's "clear a *deployable* net-of-honest-cost SR AND add to the MOM+DM book via near-zero correlation."
A modest standalone SR that diversifies can still be worth deploying; a high standalone SR that's actually a
look-ahead/cost artifact is not.

**3. HONEST PRIORS (what we already learned — do not relearn the hard way).**
- **Naive reversal ≈ 0 net.** Fixed-threshold RSI2 / raw 5-day reversal does NOT survive the honest daily engine
  (look-ahead stripped + realistic costs). The edge is in the *construction*, not the raw signal.
- **In the LIQUID universe reversion is MARGINAL** — best ~**0.2–0.3 net** once honest ([[reversion-catalog-synthesis]],
  the 11-article + stat-arb review). It is a **diversifier, not a solo star.** Be sceptical of any >0.5 standalone —
  check the engine first.
- **The construction that survived = "QP" (quality-of-reversion):**
  (a) **RARITY** — per-stock z-score of the recent move vs its *own* trailing history (adaptive oversold), not a
  fixed threshold; (b) **TREND GATE (load-bearing)** — long rare DROPS only in an UPTREND (>200d SMA), short rare
  POPS only in a DOWNTREND (buy dips in winners, fade rips in losers); (c) **EVENT HOLD** — enter only on the rare
  trigger, hold H days → few names fire/day → turnover-efficient.
- **It is SYMMETRIC and BREADTH-driven** — no bimodal winner/loser trick (that's momentum's world); many small
  independent bets, raw return target is fine (significance-weighting / tval is a MOM tool, NOT needed here — the
  opposite of the MOM sleeve). Conviction concentration HURTS reversion; breadth helps.
- **DD lever = REGIME FILTER, not stops.** Reversion *breaks* in high-vol / crisis (the "falling knife" regime) —
  turning the sleeve off on a vol/credit-stress signal is the drawdown control; hard stops are not.

**4. Honest engine is non-negotiable (this is where naive reversion dies).** signal[d] → trade at close[d+1] →
earn from d+2 (**lag=2**); per-side bps charged on **daily** turnover; dollar-neutral gross 2.0; universe filtered
for tradability (min price, min ADV). Any reversion result on a same-day or lag-0 basis is a look-ahead mirage.

---

## THE CORE DESIGN (concrete QP framework from the articles — the T1 baseline to test honestly)
- **Signal — QP rarity:** percentile of the stock's k-day return (k≈3) within its OWN trailing distribution
  (~5-yr / long lookback), scaled 0–100; low = rarer. Fire on **QP < 15** (adaptive oversold; replaces fixed RSI2).
- **Trend gate (load-bearing):** LONG rare DROPS only if `close > 200d SMA`; SHORT rare POPS only if `close < 200d SMA`.
- **Exit — event, not fixed hold:** exit at next open when `close > yesterday's high` (reversion touched); backstop
  with a time limit (~6 bars) and a stop (−5%). Few names fire/day → turnover-efficient.
- **Universe/sizing:** liquid tier (`hub.elig('liquid')`) + a relaxed tier for the capacity/decay curve; price>$1;
  position ≤ 5% of 3-month ADV; **short leg sized ~15–20% of long** (short reversion is riskier — squeeze/borrow).
- **Construction:** dollar-neutral, breadth-weighted (many small equal bets); ≤~20 long / ~20 short concurrent.
- **ML overlay (T-later, from article 2):** XGBoost P(bounce ≤5d) as a FILTER on top of QP<15 (enter iff P>~0.6).
  Features = ROC multi-h, RSI, QPI-windows, IBS, norm-ATR, dist-200SMA, turnover(cs+ts), Hurst — cross-sec standardised.
- **Regime overlay (DD lever, from article 3):** VIX filter (bear when `VIX > 15d-SMA × 1.15`) → scale the sleeve down;
  reuse STATE stress `feat` as the cleaner in-house version.
- **Engine:** honest daily — `hub.backtest_daily(W, H, lag=2, cost_bps)` (per-side bps on daily turnover, borrow).
  Any lag-0 / same-day result is a look-ahead mirage.

---

## PLAN — first tests (DEFINE)
- **T1 — honest QP baseline.** Reproduce rarity + trend-gate + event-hold on the honest daily engine. Q: does it
  beat naive reversal (~0) and clear a *deployable* net bar? Report net SR, maxDD, turnover, and **corr to DM/MOM**.
- **T2 — is the TREND GATE load-bearing?** Ablate the gate (unconditional reversion vs gated). Confirms the prior.
- **T3 — horizon & rarity sweep.** k (lookback) × H (hold) grid; find the net-SR-optimal short horizon.
- **T4 — universe / capacity.** liquid vs relaxed; does the edge live in less-liquid names (capacity limit)?
- **T5 — regime filter as DD lever.** STATE stress-off overlay vs raw; does it cut maxDD without killing SR?
- **T6 — orthogonality & book value.** ERC-add reversion to MOM+DM; does the blended SR/Sortino improve despite a
  modest standalone SR? (the actual deployment test — diversification is the whole case.)

## LOG

### T1 — 2026-07-18 · Honest QP baseline (rarity + 200SMA gate + event-hold) — ★ marginal (0.44), confirms the ceiling caveat
- **Q:** does the QP framework (QP<15 + 200d-SMA gate + 6-bar hold) clear a deployable net bar on the HONEST daily engine?
- **Method:** `mr_research.py` — QP = trailing-z→Gaussian-CDF rarity of the 3-day return (proxy for the empirical-percentile
  QP), 200d-SMA gate, fixed 6-bar hold (approx of the mean-touch exit), dollar-neutral, liquid tier, honest engine
  (lag=2, 5bp/side, borrow), short leg 50% of long. Full history.
- **Result (net SR / ann / maxDD / turn):** QP L/S gated **0.44 / 7.8% / −51% / 190×** ; QP long-only 0.42 / 11.3% / −67% / 128×.
- **Conclusion:** **MARGINAL — ~0.44, in the honest 0.2–0.4 band, FAR below Quantitativo's 1.1–1.55** (their 2bp/with-2020/
  own-engine ceiling, exactly as the MANDATORY-READING caveat said). Two immediate levers before any verdict: (1) the −51/−67%
  DD screams for the **regime off-switch** (VIX/STATE stress — not yet in); (2) **turnover 190× is a capacity/cost problem** —
  the mean-touch event exit + fewer/breadth-weighted names should cut it. NOT yet deployable; NOT yet a fair test of the
  framework (scaffold uses z-score QP proxy + fixed-hold + no ML filter + no regime overlay). Next: T2 gate ablation, T5 regime.

## REJECTED / PRIORS (do not re-test naively)
- Naive fixed-threshold RSI2 / raw 5-day reversal (≈0 net honest). Same-day / lag-0 execution (look-ahead).
- Hard stops as the DD lever (regime filter wins). Conviction-concentration (breadth wins for reversion).
- Significance-weighting / tval target (that's a MOM denoiser; reversion's raw target is fine — different problem).
- Treating reversion as a standalone high-SR star (it's a ~0-corr DIVERSIFIER; judge it in the blend).

## OPEN QUESTIONS
- Where exactly does the liquid-universe edge sit on the price/ADV spectrum (capacity vs decay)?
- Best regime signal for the off-switch (vol vs credit vs dispersion) — reuse STATE `feat`?
- Cross-sectional (rank of rarity) vs event-trigger (fire on threshold) construction — which nets more after cost?

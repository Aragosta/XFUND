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
   QP<15 AND P>60%). **CORRECTION (T2, verified from the article):** the ML FILTER alone lifts SR only **0.69→0.92**
   (avg +0.76%/trade, DD STILL >50% — the author himself says "too much"); the **1.33** figure is the *full long-only
   strategy WITH stop-loss + leverage* (45.6% ann), NOT the ML's marginal contribution. Trained on the QP<15 subset only
   (~1.2M pts), 15-yr walk-forward. So the ML edge is SMALL even in their favourable 2bp/with-2020 engine. ★MANDATORY
3. **"Long and short mean reversion machine"** (quantitativo.com/p/long-and-short-mean-reversion-machine) — the L/S build:
   short = mirror (rare POP in downtrend), **short sized ~15–20% of long** (short is riskier), **VIX regime filter** (bear
   when VIX>15d-SMA×1.15 = 90th pct → cut long exposure 1.1x→0.1x), ≤20+20 names, pos ≤5% of 3m ADV, Russell-3000 PIT.
   SR 1.55 / DD 19% (2014–24, WITH 2020). ★MANDATORY
4. **Guijarro-Ordóñez, Pelger & Zanotti (2021), "Deep Learning Statistical Arbitrage"** (arXiv 2106.04028; PDF in
   `research/papers/mr/2106.04028_DeepLearningStatArb_Pelger2021.pdf`) — the SOTA blueprint and the CORRECTION to the
   Quantitativo approach. THREE lessons that reframe the whole sleeve: (a) **residualize on a FACTOR MODEL, not raw price** —
   strip 5 PCA / **IPCA (conditional, 46 characteristics)** factors; the residual is the mean-reverting object. IPCA residual
   → OOS Sharpe **4.2** vs PCA lower. (b) **the SIGNAL is the residual's own time-series** (cumulative residual → OU **s-score**
   = how stretched from equilibrium NOW), NOT a forward-return prediction — this SIDESTEPS the sign-prediction wall (our AUC 0.50).
   Benchmark ladder: OU s-score ~1.0 → FFT+FFN ~2.1 → CNN-Transformer ~4.2. (c) works BEST on **~550 LIQUID names** (minimal
   friction) — reversion is NOT a microcap-only edge; the residual/law-of-one-price arb lives in liquid names. Half the Sharpe
   persists at 1-week hold. ★MANDATORY — this is the direction (T10 implements A+B: PCA residual + OU s-score).

**HONEST CAVEAT (do not skip):** these are Quantitativo's numbers at **2 bp, their engine, WITH COVID-2020** (removing 2020
drops the flagship to ~26% ann). Our own honest-engine review ([[reversion-catalog-synthesis]]) found reversion nets only
**~0.2–0.3 after look-ahead+cost stripping in the liquid universe.** Treat 1.1–1.55 as a CEILING; the deployable question is
the honest daily engine (lag=2, per-side bps). The STRUCTURE is the keeper; the SR is not.

---

## THE FRAMEWORK — reversion is a CLASSIFICATION (mixture-separation) problem, not a rule
*(This is the NEW direction from the QP-ML articles. Prove it before coding — Phase 1 UNDERSTAND.)*

> **⚠️ STATUS 2026-07-18 — framework TESTED (T2–T5); the ML core failed on RAW targets but the RESIDUAL target is the real lead.**
> (1) The article's binary bounce classifier on the RAW 5-day return is a coin-flip (OOS AUC 0.498; 15-yr data → 0.502) and
> NO raw/signed target transform (incl. speed-weighted) carries IC (T4). BUT (2) **residualizing the target à la Da-Liu-Schaumburg
> flips IC from −0.017 (raw) to +0.026 (market/sector-neutral idiosyncratic reversion) — T5.** The right MR target is the
> IDIOSYNCRATIC forward reversion, not the raw return (the fundamental/industry drift actively pollutes the raw signal). (3) The
> pure QP EVENT RULE is real GROSS (SR 0.68) but the binding constraint is TURNOVER/COST (≈90×/yr → net negative honest); DD
> lever = STATE regime gate. **Live direction: build the residual-target book (residual features + sector-neutral construction +
> news proxy) and test whether IC +0.026 nets positive after honest cost.** The "mixture-separation → RET" reframe below holds
> only in RESIDUAL space; do not target raw returns.

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

## HAN'S BIMODAL DISTRIBUTION applied to MR — MR eats BOTH tails (2026-07-24)

Han (2022, the MOM paper) documents that momentum-conditioned forward returns are U-shaped **bimodal**: a **winner** is
mostly high-return but **2nd-most-likely a crash**; a **loser** is mostly low-return but has **significant squeeze mass**
(more evident among losers). This is the key risk lens for MR, because **MR trades BOTH legs of that bimodal distribution:**
- **MR long leg = recent losers** → adverse mode is the loser that **keeps falling (falling knife)**. This is exactly our
  existing **KNIFE-TAIL VETO** (T13/T14, the deployable MR config) — so MR *already* handles one bimodal tail.
- **MR short leg = recent winners** → adverse mode is the winner that **keeps ripping / squeezes**. This is the tail we do
  NOT yet veto, and it is the proven-highest-value one ([[short-vol-thesis-proven]]: capping squeeze flipped a short book SR
  0.28→1.69; corr(short PnL, squeeze rate) −0.76).

**So MR is the sleeve most exposed to the bimodality (both legs), and its short-winner leg has an un-vetoed squeeze tail.**
The fix is NOT a directional tail-signal in the alpha model (reading the tail as alpha failed on MOM T23 and on MR — the
5-day bounce label is unpredictable, T3). It is a **direction-aware RISK veto in the shared CONSTRUCT layer** (peer of the
liquidity sizing): SHORT a name → down-weight high **squeeze**-propensity; LONG a name → down-weight **falling-knife**
propensity (generalize the T14 veto). Sourced from ivol/upvol/MAX5/amihud/hi52 ([[asymmetric-feature-map]]) and/or the
model's predicted-distribution **spread** (2nd moment, ~2700× more predictable — [[target-snr-return-vs-secondmoment]]).
Aggregate squeeze/crash **regime** stays in STATE (VIX→squeeze t+8.5, [[state-layer]]).

**PLANNED (MECE, one axis):** build the shared tail-veto into CONSTRUCT and A/B veto-vs-no-veto on the MR L/S book — the
**squeeze-veto on the short-winner leg** is the primary arm (proven-adjacent, un-vetoed today); the **knife-veto on the
long-loser leg** = generalizing T14 into the shared layer. This unifies MR's T14 knife-tail work + MOM's crash + the short
squeeze into ONE shared, direction-aware CONSTRUCT veto, rather than three per-sleeve hacks. See MOM_research.md "HAN'S
BIMODAL DISTRIBUTION" for the cross-sleeve framing.

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

### T2 — 2026-07-18 · FAITHFUL article build (QP-event + binary ML filter + event sim + STATE gate) — ★ ML filter is a DEAD classifier
- **Q:** implement the "Long & Short MR Machine" AS ITS ACTUAL FRAMEWORK (not the DM/MOM cross-sec ranker): QP<15/>85 EVENT
  trigger → binary ML `P(bounce≤5d)>0.60` FILTER on the triggered subset (separate long/short models) → EVENT position sim
  (enter on trigger+filter, exit on FIRST of close>prev-high | 6-bar | −5% stop, ≤20+20 names by P) → short 0.2× long,
  long 1.1×/0.1× by regime → REGIME GATE from the STATE layer (`surprise`>PIT-expanding-90th-pct = bear, replaces VIX).
- **Method:** `mr_research.py` (rewritten; event sim = `event_book`, faithful to article; STATE gate = `state_regime`).
  Liquid tier, 3295 names, OOS via BACKTEST.walk_forward (train=1512,test=252,lag=1, tiered cost+borrow), honest PnL grid.
- **Result (SR / ann / maxDD / turn):**
  - GROSS: QP-event rule **0.68 / +15.0% / −47% / 8989×**  ·  QP+ML filter **0.45 / +8.4% / −69% / 12343×**  (ML filter LOWERS gross)
  - NET (tiered): QP rule **−1.27**, QP+ML **−2.10**, QP+ML+STATE-gate **−2.12**  (all deeply negative — 90× turnover × tiered cost)
- **Conclusion:** the QP EVENT signal is REAL gross (0.68, no sign/timing bug), but (a) **the ML filter HURTS gross (0.68→0.45)
  and raises turnover**, and (b) **honest tiered cost at ~90× turnover annihilates the whole thing** (net −30 to −44%/yr →
  −100% cum). Both were pre-warned by [[reversion-catalog-synthesis]] (concentrated ≤20-name book is turnover-bound; the
  article's 1.55 is a 2bp/with-2020/net-long ceiling). The STATE gate barely moves it (bear=6.6% of days; cost, not
  regime, is the killer). Faithful ≠ deployable here.

### T3 — 2026-07-18 · WHY does ML hurt? (`MODE=auc` diagnostic) — ★ the 5-day bounce label is UNPREDICTABLE on our data
- **Q:** user's challenge — "ML can't HURT, that makes no sense." Is the classifier buggy, or genuinely sk-less?
- **Method:** `mr_research.py MODE=auc` — strict-OOS AUC + IC(P,fwd) of the SAME long bounce classifier (`_train_side`,
  train QP<15 subset, label 1[fwd>0]), per yearly walk-forward block 2010–2025.
- **Result:** **mean OOS AUC = 0.498, mean IC = −0.020, base bounce-rate 55.4%.** Per-block AUC wanders 0.44–0.55 with no
  persistence (2021 peak 0.55, 2010/2025 anti-signal 0.44). Coin-flip, faintly NEGATIVE.
- **Conclusion (answers the challenge):** NOT a bug — the classifier has ~zero skill on the 5-day directional bounce (matches
  [[target-snr-return-vs-secondmoment]] "directional return is a wall"). A sk-less filter used as a SELECTOR strictly hurts:
  it discards the QP-rarity ordering that carries the gross edge (0.68) and reshuffles by noise → lower gross + more turnover.
  ML only helps when it has skill; here it has none. Consistent with the article's OWN small lift (0.69→0.92, corrected T2).
- **Before declaring ML dead (fair Phase-2 follow-ups, NOT yet run):** (1) 15-yr train window (article's; mine=6yr) — may lift
  AUC 0.50→~0.52; (2) add the **Hurst exponent** (the article's one direct mean-reversion feature, currently MISSING from my
  set); (3) use P as a soft VETO that preserves the rarity ordering, not as the ranker. If AUC stays ≤0.52 → ML overlay REJECT,
  keep the pure QP event rule + fix the real problem (turnover/cost via broader/liquid book + longer hold).

### T3b — 2026-07-18 · "try MORE DATA" (15-yr train, article's window) — ★ does NOT rescue the classifier
- **Method:** `MODE=auc TRAIN=3780` (~15yr, ~900k oversold samples ≈ article's 1.2M), OOS 2019–2025.
- **Result:** mean AUC **0.502** (vs 0.498 at 6yr), IC −0.015. A whisker above coin-flip; still no persistence.
- **Conclusion:** sample size is NOT the constraint — the 5-day bounce sign is unpredictable regardless of data quantity.

### T4 — 2026-07-18 · TARGET transforms + reversion-SPEED target (user ask) — ★ every target IC≈0; speed is WORST on PnL
- **Q:** the sign is a wall (T3) — does a SPEED-WEIGHTED or transformed target carry IC (2nd-moment is predictable;
  Novy-Marx vol→fast reversal)? Rank trades "more desirable when reversion speed is high."
- **Method:** `MODE=target` — XGBRegressor on QP<15 subset vs 4 targets; report OOS IC vs own target AND vs plain 5d
  return (the PnL yardstick — a speed target must WIN this, not just its own definition).
- **Result (IC pred,target / IC pred,ret5):** ret5 −0.017/−0.017 · **speed λ=0.7 −0.019/−0.020** · snap1 +0.005/−0.016 ·
  sqrt-signed −0.019/−0.019. ALL ≈0, most faintly NEGATIVE; speed-weighted is the WORST on the PnL yardstick.
- **Conclusion:** no target transform rescues it. MECHANISM: a speed-weighted RETURN `Σλ^d·r_{t+d}` is still SIGNED, so it
  inherits the unpredictable direction; vol/magnitude predictability does NOT transfer to a signed target (the [[reversion-qp-sleeve]]
  entanglement). **The ML/target lever is DEAD for MR** (proven 3 ways: AUC 0.50, +data 0.502, all-targets IC 0). The gross
  edge (0.68) is the QP RARITY SORT itself, not a learnable conditional expectation — same lesson as [[reversion-catalog-synthesis]]
  ("architecture-over-rule is a MOMENTUM truth; reversion's edge is the event-rule"). LAST ML stone = add Hurst (article's one
  feature I lack), prior LOW. REAL lever = turnover/cost (90×→net<0): broader breadth (top-1000/1500 not ≤20) + longer hold +
  liquid tier + STATE DD-gate. Next: T5 turnover/breadth sweep on the PURE QP rule (no ML).

### T5 — 2026-07-18 · RESIDUAL target (Da-Liu-Schaumburg) — ★★ BREAKTHROUGH: residualizing FLIPS IC from −0.017 to +0.026
- **Q:** T4 killed all RAW signed-return targets (IC≈0/neg). Da, Liu & Schaumburg (2014, Mgmt Sci) "A Closer Look at the
  Short-Term Return Reversal": standard reversal = 4 components, only the RESIDUAL (non-fundamental/idiosyncratic) reverts;
  stripping the fundamental part → 4× risk-adj return. Avellaneda-Lee (2010): idiosyncratic residual is the mean-reverting
  object. HYPOTHESIS: the raw fwd return is polluted by unpredictable industry/market drift; residualize the TARGET → IC turns
  positive & correctly-signed.
- **Method:** `MODE=target` — XGBRegressor on QP<15 subset vs residualized targets (strip market / strip sector / risk-adj),
  vs raw. OOS IC vs own target (= neutralized-book PnL yardstick) and vs raw ret5.
- **Result (IC pred,target / IC pred,ret5):** raw −0.017/−0.017 · speed −0.019/−0.020 · **resid-XSEC +0.0257/+0.0149** ·
  **resid-SECTOR +0.0225/+0.0178** · resid-SECTOR/rvol +0.012/+0.012 · retrace-frac −0.002/−0.015.
- **Conclusion (CORRECTS T4's over-claim):** ML is NOT dead for MR — it was dead for RAW/SIGNED targets. **Residualizing the
  target flips IC from perversely-negative (−0.017) to positive & theory-consistent (+0.023–0.026)**, and it even improves
  ranking of the RAW PnL (+0.015–0.018) → the fundamental/industry drift was actively HURTING the raw signal (DLS exactly).
  The right MR target = **idiosyncratic (sector/market-residual) forward reversion**, not raw return. IC +0.026 is small but
  correctly-signed and breadth-friendly (reversion Sharpe≈IC×√breadth). NEXT: (a) residualize the FEATURES too (recent move →
  idiosyncratic move; DLS/Avellaneda-Lee residualize the SIGNAL, not just the target); (b) add a fundamental-news proxy
  (earnings/analyst-revision) to strip DLS component 3 for the 4× effect; (c) build the sector-neutral residual-target book
  end-to-end through BACKTEST — does +0.026 IC net positive after honest cost? (the real bar; turnover still the enemy).

### T6 — 2026-07-18/19 · Residual-target BOOK end-to-end + speed validation + STATE-vs-VIX gate — ★ resid target converts GROSS, net still cost-killed
- **Q:** does T5's residual-target IC (+0.026) convert to net SR through the honest engine? validate vol→speed; proof STATE gate vs VIX.
- **Method:** `mr_research.py MODE=speed` and `MODE=resid` (new `make_resid_signal`: XGBRegressor on residual fwd-reversion,
  sector-neutral construction; event-only long-only vs full-univ L/S; STATE-`surprise` vs DATAHUB-`vix` gross gate).
- **SPEED (`MODE=speed`):** oversold events by rvol quintile → reversion MAGNITUDE monotone Q0 0.39%→Q4 0.87% (2.2×); "days-to-rev"
  saturated (~3d all buckets — close>prev-high is trivially met). ⇒ vol ranks reversion SIZE (Novy-Marx partial), not clean speed;
  use vol as selection/sizing input (but it co-loads squeeze tail — the [[reversion-qp-sleeve]] entanglement).
- **BOOK (`MODE=resid`):**
  - Long-only oversold: RAW gross 0.73 · RESID gross 0.69 · **RESID net 0.16** (turn 25×) — lands on the prior honest 0.2–0.3 band.
  - Full-univ L/S sector-neut (Avellaneda-Lee): **RAW gross 0.37 → RESID gross 0.46** (resid target WINS at book level, lower turn
    41 vs 47×, DD only −27%) — T5 IC finding CONFIRMS end-to-end. BUT **RESID net −2.29** (41× turn × tiered small-name cost ≈ 30%/yr).
  - **STATE-surprise gate net −2.27 BEATS VIX gate −2.44** (STATE marginally > raw VIX as the driver) — but NEITHER rescues it;
    the −99% is a COST grind, not a stress spike (gate fires 6.6–10% of days). Cost, not regime, is the killer.
- **VERDICT:** the RESIDUAL target is the RIGHT target — real, theory-grounded (DLS), confirmed 3 ways (IC, long-only gross,
  L/S gross 0.37→0.46). STATE gate ≥ VIX gate (keep STATE). BUT the better target does NOT change the deployability verdict:
  gross too low (0.46) to clear honest cost at 25–47× turnover in the LIQUID universe → net 0.16 best, all L/S net negative.
  Exactly [[reversion-catalog-synthesis]]: reversion isn't sleeve-worthy in liquid names honestly-costed (edge lives in microcaps).
  The binding constraint is TURNOVER/COST, unchanged by the target. REMAINING paths: (a) turnover collapse (hold/hysteresis) —
  decays the fast edge; (b) microcap universe — uninvestable at scale; (c) use MR as a ~0-corr OVERLAY/tilt on MOM/DM, not standalone.

### T7 — 2026-07-19 · ★★ COST-MODEL UNIT BUG FIX — all prior NET numbers (T2/T6) were overcharged ~5×
- **Bug (user caught it — "gross 0.46 → net −2.29 makes no sense"):** `run()` fed `hub.adv_d` (DAILY avg $-volume) into
  `tiered_transaction_costs`/`tiered_borrow_fees`, whose tiers are calibrated on MONTHLY $-volume ($1B/mo→5bp … $1M/mo→60bp).
  Every OTHER sleeve feeds `hub.mdv` (monthly). Feeding daily understates liquidity ~21× → a $5M/day (eligible) name was
  charged 60bp instead of 10bp (~6× on traded names). At 25–41× turnover this fabricated a ~20–30%/yr phantom cost.
- **Fix:** `dv = hub.adv_d * 21` (daily-avg → monthly-equivalent) before the tier lookup. GROSS/IC/AUC unaffected (cost-free).
- **CORRECTED nets (MODE=resid, was → now):** long-only RESID **0.16 → 0.53** (ann 10.6%, DD −52%, turn 25×) · full-univ L/S
  RESID neut **−2.29 → −0.27** (turn 41×) · +STATE gate −0.27 · +VIX gate −0.30 (**STATE ≥ VIX confirmed**). Article NET arms
  (T2 −1.27 etc.) were likewise overcharged — re-run before quoting. **Revised verdict:** long-only residual reversion nets
  ~0.53 honest = a REAL ~0-corr diversifier candidate (not the −1.3 disaster); L/S neutral still slightly negative (turnover
  41× is the enemy → concentrate/hold). The DLS residual target + correct costs move MR from "dead" to "deployable-as-overlay".

### T8 — 2026-07-19 · Regime/VOL gate on the NET-LONG book (fix: gate is inert on a neutral book) — ★ VIX gate = DD tool not Sharpe tool
- **Q (user):** "a VIX/vol gate should increase profitability." Earlier gate test was on the L/S NEUTRAL book where a gross
  scaler is mathematically INERT (scales pnl & vol equally → SR unchanged). Re-test on the LONG-ONLY (net-long) book where a
  gate can dodge crashes. Also proof VIX vs STATE-surprise vs STATE-vol as the driver.
- **Result (long-only RESID, net):** no-gate SR 0.53 / DD −52% · **+VIX gate 0.46 / DD −33%** · +STATE-surprise 0.49 / −52% ·
  +vol(sig_port) 0.37 / −52%.
- **Conclusion:** the **VIX gate CUTS DD hard (−52→−33%)** — the article's direction — but LOWERS Sharpe (cuts return too) →
  on our data the gate is a **drawdown/Calmar tool, not a Sharpe tool.** Weaker than the article's 50→19% because our
  liquid-universe reversion DD is a slow GRIND (trending-regime breakdown), not one acute VIX crash ([[reversion-qp-sleeve]]);
  VIX only catches the acute part. **NEW: raw VIX > STATE-surprise as the gate FOR THIS SLEEVE** (VIX moved DD, surprise didn't
  — reversion DD aligns with VIX level). Keep VIX for MR crash-timing (STATE wins elsewhere). Profitability upside is NOT the
  gate; it's IC/return levers → DLS news-veto + residual features (next).

### T9 — 2026-07-19 · Vol as a RETURN DRIVER (tilt to high-vol) + residual FEATURES — ★ vol tilt hurts risk-adj; resid features help IC
- **Q (user):** "vol isn't a gate — high vol = higher returns, tilt into it." Test high-vol selection on the long-only residual book.
- **Result (long-only RESID net):** all-vol **0.53** · vol≥50pct 0.48 · vol≥70pct 0.33 · vol≥80pct 0.42 · vol≥70+VIX 0.22.
  Residual FEATURES (`RESID_FEAT=1`, MODE=target): resid-XSEC IC vs ret5 **0.0149→0.0227 (+50%)**, resid-SECTOR 0.0178→0.0251.
- **Conclusion:** high-vol names DO revert 2.2× bigger GROSS (T6 speed) but risk-adjusted it HURTS — tilting to high-vol lowers
  SR (0.53→0.33) and worsens DD (−52→−58%): the bigger move is a RISK PREMIUM offset by fatter squeeze tails + wider spreads +
  lost breadth (the [[reversion-qp-sleeve]] vol-tail entanglement). Vol is a return driver, NOT a Sharpe enhancer → keep the
  BROAD book, don't concentrate; use vol (if at all) for tail-DOWN sizing, not tilt-up. **Residual FEATURES DO help (+50% ret5-IC)
  → adopt RESID_FEAT.** BEST long-only config: residual target + residual features + broad (all-vol) + optional VIX gate for DD =
  net ~0.53. NEXT/real payoff: blend this ~0-corr sleeve into MOM+DM (standalone 0.53 understates its book value).

### T10 — 2026-07-19 · Avellaneda-Lee PCA-residual OU s-score (SOTA A+B, no end-to-end) — ★ gross 0.53 / DD −23%, net still turnover-walled
- **Q:** implement the Pelger-blueprint construction we were missing: strip K PCA factors → residual returns → OU s-score
  (dislocation), trade s<−1.25 long / s>+1.25 short among FAST reverters (κ>8.4 ⇒ rev<30d). NO forward-return prediction.
- **Method:** `mr_research.py MODE=statarb` — `make_statarb_signal`/`_sscore`; rolling 252d PCA (SVD) residual, vectorized OU
  s-score, dollar-neutral, hold 5d, sign vs conviction (∝−s) weighting; OOS via BACKTEST (lag=1, tiered).
- **Result (SR/DD/turn):** sign gross K5 0.37 / K10 0.37 / K15 0.22 / K25 0.26. **Conviction K=15 gross 0.53 / DD −23% / 35×.**
  NET: sign −1.01, conviction −0.51.
- **Conclusion:** the s-score construction WORKS gross (**conviction 0.53, DD only −23%** — much cleaner DD than the QP long-only
  −52%) and sidesteps the prediction wall (no AUC dependence). Conviction ≫ sign; ~10-15 factors best. This is the CLASSICAL
  PCA+OU tier (Pelger's ~1.0 benchmark, de-rated to 0.53 by honest tiered cost + our era). NET still negative (−0.51) — SAME
  turnover wall (35×). The remaining LEVERS to climb Pelger's ladder (0.53→higher): (a) **IPCA/conditional residuals** (their
  4.2 came from characteristics-conditioned factors, not vanilla PCA); (b) turnover control (longer hold / hysteresis / entry
  bands); (c) the s-score signal is DIFFERENT from the residual-target ML book → likely ~0 corr → ENSEMBLE the two. Not yet
  deployable alone; best DD profile we've seen. NEXT: IPCA residual + turnover reduction, then corr/ensemble vs the 0.53 QP book.

### T11 — 2026-07-19 · stat-arb refinements (OU-window / band / liquidity) + Pelger turnover mechanism — ★ OU-window is the lever
- **OU-window fix (why 0.53 not 1.0):** AL fit OU on ~60d, I mistakenly used 252d (over-smooths). Fixing to M=60 lifted gross
  **0.53→0.79** (K=10 → **0.81**), DD −15%. Most of the "0.53 vs their 1.0" gap was this bug, NOT decay. `MODE=statarb M=60`.
- **AL entry/exit BANDS** (open |s|>1.25, hold until |s|<sclose): best gross at AL's sclose=0.5 (0.66); tighter 0.05 cuts DD to
  −8.6% & turnover but slightly lower gross. Band DD is excellent (−9 to −15%) but net still <0 (turnover ~31×).
- **LIQUIDITY concentration FAILS (contra Pelger, pro our prior):** top-1000/500/300 gross 0.34/0.18/0.21 (vs 0.61 all) — the
  edge lives in the BROADER/smaller universe, NOT liquid mega-caps; and turnover is STRUCTURAL (didn't drop with concentration).
- **Pelger's turnover mechanism = cost-in-objective, END-TO-END** (5bp·‖Δw‖₁+1bp short penalty INSIDE the Sharpe loss → the net
  LEARNS to trade less). We deferred end-to-end; the 2nd-stage analog is Gârleanu-Pedersen partial-trading — but that's what our
  **DynTrad execution layer already does**, so we do NOT re-implement it in the signal. Frictions: they use 5bp flat on liquid
  names; ours is tiered 25-60bp where our edge lives. `research/papers/mr/2106.04028_...pdf` (now MANDATORY reading #4).

### T12 — 2026-07-19 · SIGNAL-LEVEL turnover cut (EWMA the alpha) — ★★ the real fix: turnover was NOISE
- **Q (user):** other ways to cut turnover INSIDE the ML framework (not execution — DynTrad already handles that)?
- **Method:** `MODE=resid` — EWMA-smooth the XGBoost residual-alpha over `smooth` rebalances before ranking (stabler ranks).
- **Result (net):** long-only smooth=1 **0.53 @ turn 25×** → smooth=3 **0.50 @ turn 7×** (ann UP 10.6→10.9%) → smooth=6 0.52 @ 7×.
  Full-univ L/S smooth=1 **−0.27 @ 41×** → smooth=3 **+0.11 @ 26×** (DD −50→−34%). **Turnover was mostly noise churn.**
- **Conclusion:** EWMA-smoothing the alpha cuts long-only turnover **25×→7× with ZERO Sharpe loss** and FLIPS the L/S book
  −0.27→+0.11. This is the in-ML turnover lever (compounds with DynTrad). ADOPT `smooth≈3`. Other in-ML levers (untested): slower
  target horizon, slower features, seed-ensemble denoising.

### T13 — 2026-07-19 · TRADE ANATOMY + forward-return DISTRIBUTION (h=1..7) — ★ the sleeve is a knife-tail selection game
- **`MODE=anatomy`** (377k QP<15 long events, payoff = sector-resid 5d fwd bp). Unconditional oversold = **coin-flip, mean −27bp,
  skew −89, p1 −15.4%** → SELECTION is everything. Subdivisions:
  - MAGNITUDE: **deepest drops (Q0) are the ONLY positive bucket** (+15.7bp, IR +0.017) BUT worst tail (−23.6%) — reversion=extreme/conviction.
  - IDIOSYNCRASY: **most-idiosyncratic drops are KNIVES** (firm-news, mean −7.9bp, worst tail); systematic drops revert (+3.6bp). *Refines DLS: veto big idio drops.*
  - VOL: high-vol = WORSE residual reversion (−38 vs −4bp) + fatter tail — confirms T9 vol-tilt-hurts at trade level.
  - ABNORMAL VOLUME: no clean separation (needs tick data).
- **`MODE=dist`** (fwd-return distribution h=1..7): RAW center drifts UP (med 0.04%→0.56%, hit 50.6→54.3%, IR peaks ~h6 0.055) =
  reversion accrues over ~1 week; but RESIDUAL (neutral) mean is **FLAT −0.27% at every h** → most raw "bounce" is BETA/sector
  participation, not idio reversion. Residual **skew −88 to −93, kurt 8000+** (knife tail). Raw is +skewed (bounces), residual −skewed (knives).
- **SYNTHESIS:** MR trade = thin, ~1-week, breadth-driven SELECTION on a beta-stripped residual, whose entire risk is a −90-skew
  knife tail. ⇒ (a) neutralization mandatory, (b) alpha = model-selection not signal-mean, (c) DD control = KNIFE/TAIL VETO
  (most-idiosyncratic + high-vol buckets), not stops, (d) turnover discipline matters (tiny per-trade IR). Visual: MR_distribution artifact.
- **NEXT:** test the knife-tail veto (drop Q0-idiosyncrasy + Q4-vol) on the residual book — does it lift net SR + cut the −52% DD?

### T14 — 2026-07-19 · KNIFE-TAIL VETO (anatomy-driven) on the smoothed residual book — ★★ the deployable MR config
- **Q:** T13 anatomy said most-idiosyncratic-drops = knives + high-vol = fat tail. Veto both on the residual XGBoost book
  (residual target + smooth=3). Does it lift net SR + cut the −52% DD? Combines paper (residual) + our anatomy + turnover fix.
- **Method:** `MODE=resid` — veto candidates with IDP (idio-drop pct) < 0.2 (most idiosyncratic crash) and/or RVP > 0.8 (top vol).
- **Result (long-only, net):** no-veto 0.50/−52%/7× · veto-idio **0.59/−46%/4.9×** · veto-vol 0.53/−51% · **veto BOTH 0.60/−47%/4.8×**.
  L/S (net): no-veto +0.11 → veto BOTH **−0.25** (HURTS).
- **Conclusion:** ★ the knife-veto IMPROVES ALL THREE axes on the LONG book — SR 0.50→**0.60**, DD −52→−47%, turnover 7→**4.8×**;
  the IDIOSYNCRASY veto (firm-news knives) is the bigger driver, exactly as the −90-skew tail predicted. **DEPLOYABLE MR CONFIG =
  long-only · DLS sector-residual target · smooth=3 · knife-veto(idio<0.2 & vol>0.8) → net ~0.60, DD −47%, turn 4.8×** — a real
  ~0-corr diversifier (vs the −1.3 disaster we started the session with). LESSON: the veto is LONG-SPECIFIC — a knife is bad to
  BUY but good to SHORT, so vetoing symmetrically kills the L/S short leg (+0.11→−0.25). Apply veto to the long leg ONLY.
- **NEXT:** blend the 0.60 long-only sleeve into MOM+DM (corr + combined book value — the deployment test); asymmetric veto on L/S
  (long-leg veto only); then IPCA residual target (Pelger's gross lever).

### T15 — 2026-07-19 · Target-IC analysis + MR↔DM correlation (deployment test) — ★ MR is anti-momentum; DM currently BROKEN
- **Target IC (0.026) — is it "wrong"?** NO. Literature: real cross-sectional IC is **0.02–0.05** (a published NASDAQ ML model
  = 0.041); Sharpe comes from IC×√breadth×√periods, not high IC. Our 0.026 is likely UNDERSTATED — measured on the restricted
  QP<15 oversold SUBSET + pooled (not full-cross-section, per-day). Levers already shown: residual features +50% (0.015→0.023);
  untested: seed-ensemble, full-cross-section per-day IC. Nothing broken about the target.
- **MR↔DM correlation (`MODE=corr`):** deployable MR long-only book (net 0.60) vs DM monthly, 2011+ overlap (131mo):
  **Pearson −0.63 · Spearman −0.56 · rolling-24m [−0.90,+0.23]** — strongly NEGATIVE, economically right (short-term reversion
  = ANTI-momentum: buy oversold losers = opposite of DM buy-winners). A strong-negative-corr positive-Sharpe sleeve is a POWERFUL
  diversifier IF DM is positive.
- **★ DISCOVERY — DM is currently BROKEN:** DM.py on the CURRENT data = net SR −0.37 / maxDD −85% (IC fine +0.040 → SIGNAL ok,
  BOOK broken), NOT the known champion ~0.95. NOT from the DATAHUB working-tree edit (that only removed an unused backtest_daily
  helper) → most likely the recent commit "Consolidate to self-contained DATAHUB; prune parquets" CHANGED THE UNDERLYING DATA;
  DM's 0.95 was on the OLD data. ⇒ the blend SR can't be trusted until DM is re-validated on current data. The corr SIGN (−0.63)
  is robust regardless. NEXT: re-validate DM on current data (separate task), OR use a self-contained 12-1 momentum proxy to
  quantify the reversion↔momentum decorrelation independent of DM.py, then the ERC blend.

### T16 — 2026-07-19 · OVERNIGHT/INTRADAY reversal decomposition (`MODE=onx`) — ★ real premium, UNTRADEABLE on our daily-close engine
- **Provenance (deep lit sweep, 2024–25):** we ASSUMED close-only data — WRONG. `hub.open_d/high_d/low_d` are populated
  through 2026-07 (nan% 57 = universe coverage, same as close). That unlocks the hottest reversal line: **Liu-Liu-Wang-Zhou-Zhu
  "Overnight-Intraday Reversal Everywhere"** (buy low past-overnight / sell high → intraday SR 2–5× close-to-close reversal),
  **Brogaard-Han-Kim 2024 "Intraday Residual Reversal"** (162% gross, liquidity provision to the TRANSITORY component),
  **Baltussen-Da-Soebhag "End-of-Day Reversal."** All = the same Nagel liquidity-provision premium, decomposed into
  OVERNIGHT (open/prev-close) vs INTRADAY (close/open) legs; the overnight leg is claimed to revert hardest.
- **Q:** does the overnight leg give a cleaner reversal signal than close-to-close ON OUR ENGINE? `MODE=onx`: decompose the
  k=3 move into ON/ID/CC legs (log, sector-residual), (A) measure per-day reversal IC vs the DLS fwd residual reversion,
  (B) run each leg as a direct reversal book (long top-N/short bottom-N, hold 10d) through the honest engine (lag=1, tiered).
- **Result (A · reversal IC all / on QP<15):** ON **0.0043 / 0.0002** · ID 0.0101 / 0.0018 · **CC 0.0120 / 0.0018**. The
  OVERNIGHT leg is the WEAKEST reversal signal; close-to-close is the STRONGEST — the OPPOSITE of the literature headline.
- **Result (B · direct reversal book, net SR / DD):** ON **−0.77 / −84%** · ID −0.09 / −52% · CC −0.24 / −68% (all raw
  no-event/no-veto/no-turnover-control books → negative as always; the RANKING is the finding). Corr to DM ≈ 0 for all legs
  (raw L/S books, much weaker than the deployable book's −0.63).
- **Conclusion (negative, decisive):** the overnight-reversal premium is REAL but **not tradeable on our daily close-to-close
  infrastructure.** The lit edge is harvested INTRADAY (enter at the open on low overnight return, exit at the close) — it
  reverts within HOURS. Our engine enters at the NEXT close and holds ~10d, by which point the overnight signal is stale and
  in fact INVERTED (SR −0.77). Capturing it needs open-execution / intraday granularity we do NOT model. The OHLC unlock is
  genuine but its value is for FEATURES (IBS / GK-vol / nATR — already in `features()`), NOT an overnight sleeve. **No change to
  the deployable book; close-to-close residual reversion (what it already uses) remains the right signal for a close-held book.**
- **Frontier note (deferred by user — no turnover work for now):** the REAL lever the 2025 lit points at is **cost-in-the-
  objective end-to-end** — Epstein-Wang-Choi-Pelger "Attention Factors for Statistical Arbitrage" (ICAIF '25, arXiv 2510.11616)
  = Pelger's DLSA follow-up, net SR **2.28** (vs DLSA 1.25, +84%) on the 500 largest US names, by folding the 5bp turnover
  penalty INTO the Sharpe objective so the factors LEARN to be low-turnover. Corroborated by Baldi-Lanfranchi "Transaction-cost-
  aware Factors" (AFA 2024, up to 2.5× net squared Sharpe, "most beneficial for high-turnover factors otherwise unprofitable")
  and de Groot-Huij-Zhou (reversal cost is concentrated in small-cap trading; trade-bands restore it). This directly attacks
  MR's ONE wall (turnover) but is TURNOVER-CONTROL work → parked per user instruction. When turnover work resumes, this is #1.

### T17 — 2026-07-19 · K-FACTOR sweep on the PCA residual ("weak factors matter"?) — ★ does NOT replicate on vanilla PCA
- **Q:** the Attention Factors ablation (Epstein-Pelger 2025) found *raising the factor count K=8→30 helps* ("weak factors that
  capture local dependency patterns are important for trading"). Our stat-arb arm uses K=10-15 PCA. Does more PCA help GROSS?
  (residual-QUALITY lever, NOT turnover.) `MODE=statarb KSWEEP=1 M=60` — all names, band exit, gross.
- **Result (gross SR):** K=5 **0.36** · K=10 **0.61** · K=15 0.51 · K=20 0.43 · K=30 0.35 · K=40 0.49. Peak at **K=10**,
  monotone DECLINE through K=30 (K=40 a noisy uptick, still ≪ K=10).
- **Conclusion (negative, sharpens the IPCA note):** "weak factors matter" does NOT transfer to VANILLA PCA on our data —
  gross peaks at K≈10 and adding PCA components 11-40 HURTS. Mechanism: PCA eigenvectors beyond ~10 are NOISE directions; using
  them to residualize OVER-strips the residual (removes real idiosyncratic reversion signal) + adds turnover. The lit finding is
  specifically about LEARNED CONDITIONAL factors (attention/IPCA characteristic-conditioned), which capture genuine structure a
  PCA eigen-decomposition cannot. ⇒ the residual-quality lever for us is NOT "more PCA factors" (rejected) but **conditional
  residuals (IPCA/attention)** — the heavier build flagged at T10/T11, now the only remaining gross lever. K=10 stays optimal for PCA.

### T18 — 2026-07-19 · #1 RMT/MP denoising + #2 model-free stationarity gate (signal/noise levers, NO turnover) — ★ both confirm the PCA ceiling
Two non-turnover signal-quality prototypes on the stat-arb arm (`MODE=statarb` flags `DENOISE=1`, `QGATE`/`QSWEEP`):
- **#1 RMT/Marchenko-Pastur factor DENOISING (`DENOISE=1`).** Instead of a fixed K, strip only eigenvectors above the MP
  noise edge λ+=(1+√(N/T))² (Laloux-Bouchaud / Lopez de Prado) — a data-driven per-window factor count that never strips
  noise. **Result (gross, MP-count cap sweep):** cap=5→count 5.0→**0.36** (matches T17 fixed-K=5 = sanity check ✓) · cap=10→
  count 9.5→**0.59** · uncapped→count **11.8**→**0.51**. The natural MP count (~11.8) lands NEAR but BELOW hand-tuned K=10
  (0.61); capping at 10 recovers 0.59. **Verdict:** MP denoising is a robust, TUNING-FREE auto-K (removes the K hyperparameter,
  lands near the optimum) but is NOT a gross lever — its count is slightly too generous (keeps 1-2 noise-ish factors). The T17
  hope ("denoising lets weak factors help → stops the K>10 decay") is NOT supported: there is no hidden signal in the higher
  eigenvectors to recover. Confirms the residual-quality ceiling is STRUCTURAL to PCA.
- **#2 model-free stationarity gate (`QSWEEP`).** Add a signal-quality filter on which residuals to trust, beyond the OU
  κ-speed gate: variance-ratio VR(5)<1, Ning-Lee empirical-mean-reversion-time (crossing-run-length, keep fast half), or both.
  **Result (gross, K=10):** kappa (baseline) **0.61** · vr 0.46 · emrt 0.55 · combo 0.59 — every model-free gate HURTS or
  matches, none beats the κ-gate, and all raise turnover. **Verdict:** the OU κ-gate ALREADY does the tradeable-reversion
  selection; VR/EMRT just drop names without adding quality (our residuals passing κ>8.4 are already stationary enough). #2 REJECT.
- **★ SYNTHESIS (decisive roadmap):** on VANILLA PCA residuals the cheap signal/noise levers are now EXHAUSTED — denoising
  (#1), factor-count (T17), stationarity gates (#2) all confirm the ceiling at gross ~0.61 (κ-gated, K≈10, M=60). There is no
  more juice in the PCA residual. The ONLY remaining GROSS lever is a BETTER mean-reverting OBJECT = **conditional / nonlinear
  residual** (autoencoder asset pricing Gu-Kelly-Xiu 2021 = nonlinear IPCA; or Box-Tiao/d'Aspremont-Cuturi predictability
  baskets), the heavy build flagged since T10/T17. KEEP: MP-denoising as the tuning-free auto-K default (robustness, not SR).

### T19 — 2026-07-19 · FUSE the two MR engines: OU s-score state as cross-sectional XGBoost features — ★ redundant, no IC lift
- **Q (user):** we have TWO MR engines — OU s-score (time-series, per-name dislocation) and XGBoost (cross-sectional rank).
  Combine them: feed the OU state (sscore + kappa + cumulative-residual-momentum, MP-denoised, PIT trailing window, stride-5
  ffill) as FEATURES into the cross-sectional residual-target book. Does the time-series view add cross-sectional IC?
- **Method:** new `statarb_state()` (cached /tmp/statarb_state.pkl) + `features()` hook behind `STATARB_FEAT=1`; measured via
  `MODE=target` IC(pred, ret5) baseline (RESID_FEAT) vs +STATARB_FEAT.
- **Result (IC pred,ret5 — baseline → fused):** resid-XSEC 0.0221→**0.0213** · resid-SECTOR 0.0168→0.0164 · resid-SECTOR/rvol
  0.0151→0.0195 (only winner) · retrace −0.011→−0.018. Flat-to-WORSE on the main targets.
- **Conclusion (negative, informative):** the OU s-score is a nonlinear function of RECENT RESIDUAL RETURNS, which XGBoost
  ALREADY extracts from its QP/ROC/RSI features → the two engines carry the SAME information viewed two ways, NOT complementary
  axes. Feature-fusion adds 3 redundant/noisy inputs → slight IC degradation. ⇒ "combine the two versions" via features is a
  DEAD END; the only sensible combine is an ENSEMBLE of the two BOOKS (likely low-corr diversification), not feature fusion.
- **★ THE REAL IC LEVER (from T13 anatomy, DM-analog):** the residual reversion MEAN is FLAT (−0.27% at every h); the entire
  structure is a −90-skew KNIFE TAIL. Regressing the residual MEAN (current book) optimizes the wrong functional — there's
  little mean to predict; the edge is WHICH NAMES ARE NOT KNIVES (tail/selection). This is EXACTLY the DM situation: DM is
  unique because it doesn't predict the momentum mean — it reframes as a MIXTURE and computes RET=Σpₖμₖ (conditional expectation
  over classified components). The MR mixture = {liquidity-bounce, information-knife} (Da-Liu-Schaumburg). ⇒ the truly-unique MR
  move = predict the DISTRIBUTION of the forward residual (HL-Gauss histogram / multi:softprob) and read out a KNIFE-PENALIZED
  functional (Σpₖμₖ with left-tail buckets down-weighted), NOT the mean. NEXT: prototype the distributional knife-aware residual
  target vs the mean-regression baseline (does IC and — more importantly — the tail-adjusted book SR improve?).

### T20 — 2026-07-19 · ★★ TARGET FORM is the IC lever — asymmetric knife-clip lifts IC 0.0168→0.0300 (+79%)
- **Q (user "try several things"):** the residual MEAN is a wall (IC ~0.017) but its RISK is a −90-skew knife tail (T13). MSE
  on that target explodes on the tail. Try target FORMS that tame the knife (DM lesson: engineer the target, don't forecast raw
  return). `MODE=target` batch (IC vs the ret5 PnL yardstick) + `MODE=hlgauss` (Stop-Regressing distributional).
- **Result (IC pred,ret5 — sector-residual target unless noted):** raw **0.0168** · winsor±10% 0.0221 · winsor±5% **0.0261** ·
  winsor±3% 0.0278 · winsor±2% 0.0281 · tanh(/5%) 0.0244 · **clipL−5% (asymmetric: floor knives at −5%, keep bounce upside)
  0.0300** · clipL−3% 0.0288 · rank(gauss) 0.0191 · sign 0.0248 · 1d-snap 0.0176 · knife-safe-binary −0.0217 (own-IC **0.18**).
- **Distributional / Stop-Regressing (`MODE=hlgauss`, 10-bucket histogram → readouts):** Σpₖμₖ (DM mean) **0.0198** · −E[bucket]
  0.0202 · P(up) 0.0117 · knife-penalized (λ=3) −0.002 (over-penalizes). Confirms the thesis DIRECTION (distributional beats
  raw-MSE 0.0168→0.020 = tail-robustness real) BUT the crude HARD-BIN version (XGBoost multi:softprob, no Gaussian soft labels)
  lands BELOW simple winsor+MSE — hard rank-bins discard magnitude resolution winsor keeps. True HL-Gauss (soft labels) needs a NN.
- **★ CONCLUSION — the IC lever is the TARGET FORM, not the model/features:** floor the sector-residual reversion target at −5%
  (asymmetric clip: knives are noise → clip; bounce upside is signal → keep) → **IC 0.0300, +79% over raw**. Beats symmetric
  winsor (0.0261) because MR's risk is ONE-SIDED (−90 skew). This is the DM-analog realized: DM's edge = engineered target
  (bimodal reclassification), not a better forward-return forecast; MR's = knife-clipped residual target, not a better feature/model.
  Two proven negatives frame it: OU-state features redundant (T19), model-free gates hurt (T18) → it was never the features/model.
  Knives themselves are HIGHLY predictable (binary own-IC 0.18) → keep them as the VETO (T14), not the ranker.
- **NEXT:** convert IC 0.0300 → net book SR through the honest engine (`make_resid_signal` with the clipL−5% target vs current
  sector-residual) — does +79% IC lift the deployable net 0.60? Then: soft HL-Gauss (NN) to see if the distributional beats winsor
  with proper label smoothing; asymmetric knife-clip is the cheap deployable win NOW.

### T21 — 2026-07-19 · STOP-REGRESSING done properly (soft HL-Gauss) + robust-LOSS — ★ the structural lever is the LOSS (L1), not a clip
- **Q (user):** the T20 asymmetric clip is "quick and dirty" (magic −5%); we want a STRUCTURAL fix like Stop-Regressing. Built the
  true soft-label HL-Gauss (custom soft-CE objective in `xgb.train`, `MODE=hlgauss SOFT=1`) + robust losses.
- **Soft HL-Gauss variants (`PROJ`):** rank-gauss (σ=1.0) **0.0222** — beats hard-bin 0.0202 (soft labels recover resolution,
  user's "rankgaus on stop-regressing" idea CONFIRMED) but symmetric → loses the asymmetry. Value-space quantile centers +
  C51 2-hot projection (STRUCTURAL, no clip, bounded CE loss): Σpₖμₖ **0.0174** (WORSE — the value-mean READOUT is tail-sensitive:
  a little prob mass on the −20% bottom-bin center swings the score; bounded LOSS ≠ bounded READOUT), P(up) 0.0199.
- **★ ROBUST LOSS (`MODELOBJ=reg:quantileerror`, RAW target, NO clip):** τ=0.3 −0.006 · τ=0.4 0.009 · **τ=0.5 (median/L1) 0.0262**.
  Median/L1 regression = **+56% over MSE (0.0168), ZERO magic numbers** — L1 doesn't square the residual so a −40% knife gives a
  BOUNDED gradient = the robustness the clip faked, as a principled objective change. Asymmetric LOSS (τ<0.5) went the WRONG way.
- **★★ THE STRUCTURAL CONCLUSION — the lever is LOSS-ROBUSTNESS, not model/features/clip:** IC ladder — MSE 0.0168 < value-soft
  0.0174 < hard-bin 0.0202 < rank-gauss-soft 0.0222 < **median/L1 0.0262** ≈ winsor±5% 0.0261 < asymmetric-clip 0.0300. EVERY
  robust loss (L1, distributional CE, winsor) beats MSE +50–80% → MSE was the disease, the knife tail dominated it. Cleanest
  STRUCTURAL cure = **L1/median loss** (principled, no threshold, matches winsor). The clip's extra 0.0038 = a ONE-SIDED economic
  prior (keep bounce upside=signal, floor knife downside=noise); it needs a threshold because "info-declines are unpredictable"
  IS a one-sided fact — and it belongs in WHAT YOU KEEP (target), NOT in the loss tilt (asymmetric quantile HURT). Distributional
  Stop-Regressing's theoretical robustness did NOT beat plain L1 here (trees + our SNR: multiclass is a noisier learner than one
  robust regression; distributional shines with more data / NNs). ADOPT L1/median loss as the structural base; clip = optional top-up.
- **NEXT:** convert (L1-loss book, and clipL book) → net SR through the honest engine (`make_resid_signal`); soft HL-Gauss only
  worth revisiting with a NN (soft labels + value readout capped = would encode asymmetry structurally). Harness: MODE=hlgauss
  SOFT=1 PROJ=value|rank, MODE=target MODELOBJ=reg:quantileerror|reg:absoluteerror|reg:pseudohubererror.

### T22 — 2026-07-19 · does IC convert to net BOOK SR? — ★ NO on the veto-book (redundant), YES once veto is OFF (L1 replaces it)
- **Q (user "if there's more signal there should be more returns"):** convert IC 0.017→0.026→0.030 (MSE→L1→clipL) to net SR
  through the honest engine (`make_resid_signal`, added `loss` + `conv` params). Champion long-only book (resid·smooth3·veto BOTH).
- **Result (net SR / ann / maxDD, veto BOTH):** MSE·resid **0.58**/10.2%/−47.6% · L1·resid **0.58**/10.0%/**−45.5%** · Huber 0.58 ·
  MSE·clipL 0.58/10.9% · L1·clipL 0.58/−45.5%. ALL round to SR 0.58 (ann/DD/turn DIFFER slightly → genuinely different books, not
  a bug) — the IC lift did NOT move SR. **WHY:** the position-level KNIFE-VETO already removes the exact names where the losses
  differ → tradeable top-150 ≈ same set → SR unchanged. Loss is REDUNDANT with veto (but L1 still trims DD −47.6→−45.5).
- **★ PROOF (veto OFF — let the LOSS handle knives):** MSE·resid·noveto **0.53** · **L1·resid·noveto 0.59** · MSE·clipL·noveto 0.51.
  Now the loss SEPARATES: L1 (0.59) beats MSE (0.53). **The robust L1 loss structurally REPLACES the ad-hoc veto** (0.59≈veto's
  0.58, no magic thresholds). So the signal IS real & DOES convert — it was masked by the veto doing the same job.
- **Conviction-weight + breadth (veto BOTH):** L1·conviction 0.56 (HURTS — equal-weight breadth wins, reversion is breadth-not-
  conviction, confirms prior) · L1·clipL·conv·N=400 0.58 (DD better −43.9%, turn lower). SR ceiling ~0.58-0.59 is STRUCTURAL
  (cost/breadth), not the target/loss/IC. IC↑ ≠ SR↑ here ([[quantile-edge-decomposition]]): the IC gain lived in the knife region
  the veto already handled. LESSON: adopt L1 loss + DROP the veto (cleaner, same SR, slightly better DD).

### T23 — 2026-07-19 · ★★ FEATURE ENGINEERING — IC nearly TRIPLES (0.0168→0.0491) via DLS info-vs-liquidity OHLC features
- **Q (user "try feature engineering, get it up"):** engineer features grounded in DLS/Nagel economics using our OHLC+volume:
  OVERNIGHT-gap vs INTRADAY-move split (on3/id3/gapfrac/gapdn = news/info=KNIFE vs order-flow/liquidity=BOUNCE — the DLS
  separator), Amihud illiquidity, high-low spread, downside-deviation, worst-drop. Behind `ENGFEAT=1` in `features()`.
- **Result (IC pred,ret5 — RESID_FEAT → +ENGFEAT):** resid-SECTOR 0.0168→**0.0263** · resid-XSEC 0.0221→**0.0344** · winsor±5%
  0.0261→**0.0359** · **clipL−5% 0.0300→0.0491** (the champion — +64%). clipL+ENGFEAT = **0.0491 ≈ 2.9× the original 0.0168 baseline.**
- **★ Why they COMPOUND with the clip:** the engineered features IDENTIFY knives (a 3-day drop that came OVERNIGHT = gap = news =
  knife; came INTRADAY = liquidity = bounce), and the asymmetric clip stops the model wasting capacity on knife MAGNITUDE →
  together they nail the knife-vs-bounce separation that IS the sleeve. This is the Da-Liu-Schaumburg info-vs-liquidity decomposition
  finally OPERATIONALIZED in the features (we had OHLC all along — see [[mr-frontier-2025]] the close-only assumption was wrong).
- **NOTE (contrast T16):** the overnight SLEEVE failed (untradeable on daily close engine) but the overnight RETURN as a FEATURE
  (info-vs-liquidity separator) is a big win — the data unlock pays off as features, exactly as T16 predicted.
- **NEXT:** does 0.049 IC convert to net SR? Book test = ENGFEAT + clipL/L1 + NO veto (features now do the knife-ID the veto did) —
  running. Then per-feature ablation (which of the 8 drove it — likely the overnight/gap separators). Champion IC config =
  clipL−5% target · RESID_FEAT+ENGFEAT · (L1 loss, veto-off in book).

### T24 — 2026-07-19 · ★★★ WHY tripled IC doesn't move net SR — the IC gain is KNIFE-side, the book is long-only (gross-level decoupling)
- **Q (user, twice: "if more signal, more returns"; "did you make a mistake?"):** the T23 features tripled IC (0.0168→0.0491) but
  the book stayed ~0.58. VERIFY it's not a bug, and LOCATE where the IC dies (gross vs net). Diagnostic `scratchpad/diag.py`.
- **VERIFIED — not a bug:** engineered features ARE in the tensor (20→28 cols: amihud/downsemi/gapdn/gapfrac/hlspread/id3/
  mindrop21/on3); eligible universe unchanged (1674/day). Features flow.
- **★ DECISIVE — the ceiling is at the GROSS level:** clipL BASE gross **0.55** → clipL+ENG gross **0.56** (net 0.52). Tripled IC
  barely moves GROSS SR. ⇒ NOT cost/turnover (that would be gross↑/net-flat), NOT a bug → a real GROSS-level IC↔SR decoupling.
- **★★ MECHANISM (the key insight of the session):** the engineered features are KNIFE DETECTORS (overnight gap-downs, downside-
  dev, worst-drop) and the clip sharpens KNIFE-RANKING → they raise IC by better ordering the BOTTOM of the cross-section. But the
  book is LONG-ONLY top-150 — it NEVER buys the bottom, so better knife-ranking doesn't change what it holds (it already avoids
  knives via the sort). IC is measured over the WHOLE cross-section; a long-only book monetizes only the TOP. The entire IC gain
  landed in a region the book doesn't trade = [[quantile-edge-decomposition]] proven cleanly (+IC ≠ +PnL when the edge is where you
  don't trade). CONFIRMS [[asymmetric-feature-map]]: the SHORT/knife side is the predictable (~50t vol/squeeze) problem; the LONG
  side is near-featureless WITHIN the top (the SR comes from the sort itself, not from discriminating among good bounces).
- **★ IMPLICATION — the MR net-SR lever is NOT signal quality (IC):** it's WHERE the signal lives. To lift a LONG-ONLY book,
  features must discriminate AMONG THE GOOD BOUNCES at the top (hard — residual near-featureless there), NOT identify knives at the
  bottom. To monetize knife-ID you must TRADE the bottom (short the knives = squeeze+borrow = the tax, [[short-leg-is-the-tax]]; or
  veto = already done). MR gross ceiling ~0.55 is set by the difficulty of ranking good bounces, not by knife-avoidance. STOP
  chasing IC via knife-features; either accept ~0.55 gross as the long-only ceiling or attack the long-side bounce-discrimination
  problem directly (measure IC WITHIN the top decile). Feature engineering "worked" (IC↑) but on the wrong side of the book.
- **NEXT:** decompose IC by rank region (top-tercile vs bottom) to confirm the gain is bottom-only; pivot any feature work to
  LONG-side bounce-quality; OR judge the sleeve done at long-only net ~0.55-0.60 (a real ~0-corr diversifier, its actual job).

### T25 — 2026-07-19 · LONG-SIDE (bounce-magnitude) features — ★ first to move GROSS (0.56→0.59), but net ceiling-bound ~0.60
- **Q (user "add more features for the long leg"):** T24 said the IC gain was knife-side (bottom), useless to a long book. Add
  LONG-SIDE features that discriminate WHICH oversold names bounce MOST: Bollinger-z (bbz20, stretch below short mean), short
  stretch (stretch10), lag-1 autocorr (acf1, negative=reverts), variance-ratio (vr5<1=reverts), volume-exhaustion (voldecel),
  spread (cs_spread). Behind `LONGFEAT=1`. Test by BOOK GROSS SR (not IC — T24 showed IC is misleading here).
- **Result (long-only clipL book, GROSS SR):** RESID+ENG (knife) 0.56 · RESID+LONG (bounce, no knife) 0.56 · **RESID+ENG+LONG 0.59**
  (ann 13.5→14.7%). ★ SYNERGY: neither feature-set alone moves gross; TOGETHER they lift it (knife features clean the bottom, long
  features sharpen the top → the top-150 selection only improves with BOTH). These are the FIRST features all session to move GROSS
  (contrast T24 knife-features: tripled IC, gross flat) — confirms the long side is where the (thin) monetizable signal is.
- **NET (RESID+ENG+LONG, clipL):** veto-OFF 0.55 · **veto-ON deployable 0.60** (ann 11.2%, DD −47%, turn 483%) = IDENTICAL to the
  original T14 champion (0.60). The gross gain (0.56→0.59) does NOT survive the veto-on/top-150/cost structure → deployable net
  stays ~0.60. Keep the long features (moved gross the right way, cost nothing) but they don't break the ceiling.
- **★★ SLEEVE VERDICT (conclusive):** pushed EVERY signal lever — target form (clip/L1), loss (robust/distributional), model
  (Stop-Regressing/HL-Gauss), fusion (OU state), features (knife + long). MR long-only is STRUCTURALLY ceiling-bound at net ~0.60
  / gross ~0.55-0.59. Signal quality (IC) is NOT the binding constraint (tripled it, no net gain). Binding constraints: (a) the long
  side is near-featureless WITHIN the top (bounce magnitude intrinsically hard, [[asymmetric-feature-map]]), (b) turnover/cost +
  top-N breadth (construction levers = the parked turnover work). MR is a real ~0-corr net-0.60 DIVERSIFIER — that's its job, not a
  standalone star. Signal-side of the sleeve = DONE. Only net-movers left: turnover-aware construction, or trade the short/knife
  side & pay the squeeze tax ([[short-leg-is-the-tax]]). Champion = clipL/L1 · RESID+ENG+LONG features · smooth3 · top-150 EW · veto-on.

## REJECTED / PRIORS (do not re-test naively)
- **Feature engineering to raise MR net SR via KNIFE-detection features — REJECT (T24).** Overnight-gap/downside/worst-drop
  features triple IC (0.049) but the gain is BOTTOM/knife-side; a long-only top-N book doesn't trade the bottom → gross SR flat
  (0.55→0.56). +IC ≠ +PnL. Long-side bounce-discrimination is the real (hard) lever, or accept the ~0.55 gross ceiling.
- **Conviction-weighting the reversion book — REJECT (T22).** Equal-weight breadth beats it (0.58 vs 0.56); reversion is
  breadth-not-conviction ([[reversion-qp-sleeve]]). Breadth N=400 helps DD/turnover, not SR.
- **Distributional Stop-Regressing (soft HL-Gauss) as the IC winner on trees — REJECT for now (T21).** Real & robust (rank-gauss
  0.0222 > hard-bin 0.0202 > MSE 0.0168) but does NOT beat plain L1/median loss (0.0262) or the asymmetric clip (0.0300) on
  XGBoost at our SNR; value-space mean readout is tail-sensitive (0.0174). Revisit only with a NN (soft labels + capped readout).
- **Asymmetric LOSS (quantile τ<0.5) — REJECT (T21).** Predicting the downside quantile ranks knives, IC goes negative
  (τ=0.3 −0.006). The one-sidedness belongs in the TARGET (clip/keep) not the loss tilt.
- **Fusing OU s-score state as FEATURES into the XGBoost residual book — REJECT (T19).** Redundant with QP/ROC (both are
  functions of recent residual returns); no IC lift, slight degradation. Combine the two engines as an ENSEMBLE OF BOOKS, not features.
- **Model-free stationarity gates (variance-ratio / EMRT) on top of the OU κ-gate — REJECT (T18).** kappa 0.61 > combo 0.59 >
  emrt 0.55 > vr 0.46; the κ-gate already selects tradeable-reversion; extra filters drop names + raise turnover, no gross gain.
- **RMT/MP denoising as a GROSS lever — REJECT (T18).** Useful as a tuning-free auto-K (~11.8 factors, gross 0.51-0.59) but
  does not beat hand-tuned K=10 (0.61); no hidden signal in eigenvectors past ~10. PCA residual-quality ceiling is structural.
- **"More PCA factors" (K>10) on the stat-arb residual — REJECT (T17).** Gross peaks at K≈10, declines to K=30. The Attention
  Factors "weak factors matter" result needs LEARNED conditional factors (IPCA/attention), not more PCA eigenvectors (which are
  noise directions beyond ~10 and over-strip the residual). Residual-quality lever = conditional factors, not higher K.
- **Overnight/intraday reversal as a SLEEVE on our engine — REJECT (T16).** Real premium (Liu et al / Brogaard-Han-Kim /
  Baltussen-Da-Soebhag) but harvested INTRADAY (open→close); our daily close-to-close engine can't capture it (overnight signal
  stale/inverted by next-close entry → ON-reversal SR −0.77). Close-to-close residual move stays the best close-held signal.
  OHLC data IS available (`hub.open_d/high_d/low_d`) — use it for FEATURES, not an overnight book.
- **Symmetric knife-veto on a L/S book — REJECT (T14).** The veto is long-specific (knife bad to buy, good to short); vetoing
  high-vol/idiosyncratic names symmetrically removes the profitable SHORT candidates (+0.11→−0.25). Long-leg veto only.
- **Vol TILT (concentrate into high-vol oversold) — REJECT (T9).** High-vol reverts bigger gross (2.2×) but risk-adj it lowers
  SR + worsens DD (squeeze tail + cost + breadth loss). Vol is a raw-return driver, not a Sharpe lever. Keep the broad book.
- **Gross-scaling regime gate on a DOLLAR-NEUTRAL book = inert** (scales pnl & vol together → SR unchanged). Gates only help a
  book with directional exposure to cut (net-long). Apply MR gates to the long/net-long book, not the neutral one (T8).
- **ML on RAW/signed reversion targets — dead (T2/T3/T4)** (AUC 0.50, IC≈0). BUT the RESIDUAL (idiosyncratic) target is ALIVE
  (T5/T6, IC +0.026, L/S gross 0.37→0.46) — the raw return's fundamental drift polluted it. Use residualized targets, NOT raw.
- **COST INPUT: always feed MONTHLY $-volume (`hub.mdv`, or `adv_d×21` on the daily grid) to the tiered schedules — NEVER raw
  `adv_d`** (T7). Daily-into-monthly-tiers overcharges ~5× and fabricates net losses.
- **Reversion as a standalone LIQUID sleeve — not deployable net (T6, confirms [[reversion-catalog-synthesis]]).** Best honest net
  0.16 (long-only); every L/S neutral book is net-negative — turnover/cost is the wall, NOT the signal (which is real gross). The
  residual target improves GROSS but not the net verdict. Judge MR as an overlay/diversifier, not a solo book.
- Full-universe residual L/S at honest tiered cost trades too many expensive small names (41× turn → ~30%/yr drag). Concentrate/hold.
- Naive fixed-threshold RSI2 / raw 5-day reversal (≈0 net honest). Same-day / lag-0 execution (look-ahead).
- Hard stops as the DD lever (regime filter wins). Conviction-concentration (breadth wins for reversion).
- Significance-weighting / tval target (that's a MOM denoiser; reversion's raw target is fine — different problem).
- Treating reversion as a standalone high-SR star (it's a ~0-corr DIVERSIFIER; judge it in the blend).

## OPEN QUESTIONS
- Where exactly does the liquid-universe edge sit on the price/ADV spectrum (capacity vs decay)?
- Best regime signal for the off-switch (vol vs credit vs dispersion) — reuse STATE `feat`?
- Cross-sectional (rank of rarity) vs event-trigger (fire on threshold) construction — which nets more after cost?

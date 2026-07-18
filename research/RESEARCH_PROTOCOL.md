# RESEARCH PROTOCOL — fool-proof pipeline for testing ideas

**Every new idea (feature, target, model, combination, execution, sleeve) passes through the same five
phases, in order, with no shortcuts.** The phases are MECE: each does exactly one job, together they
cover the whole pipeline. **Understanding comes first: if you cannot explain the idea and why it should
work in plain words, you are not ready to write a test.**

```
Phase R  READ FIRST    read the sleeve's <SLEEVE>_research.md IN FULL + its MANDATORY-READING papers  ← before ANY word/action
Phase 0  SUBSTRATE     fixed universe + data hygiene        (set once, never per-idea)
Phase 1  UNDERSTAND    explain the idea + mechanism + prior art in plain words, BEFORE any code
Phase 2  DEFINE        one hypothesis, one change, a control, a pre-registered bar
Phase 3  MEASURE       leak-free · net-of-cost · standard metrics · bagging discipline
Phase 4  ANALYZE       decision criteria + overfitting/robustness audits → verdict
```

The mantra: **READ the log+papers → full universe → UNDERSTAND the idea → clearly DEFINE → clearly MEASURE → clearly ANALYZE.**

---

## PHASE R — READ FIRST (MANDATORY, before you say or do ANYTHING on a sleeve)

**Before proposing an idea, writing a test, running code, or making ANY claim about a sleeve, you MUST first read
that sleeve's living log (`MOM_research.md` / `VALUE_research.md` / `MR_research.md`) IN FULL, and every paper listed
in its `⚠️ MANDATORY READING` section.** This is not optional and not a skim.

- **Why:** most "new ideas" and "surprising bugs" are already answered — as a fact, a REJECTED entry, or a paper's
  central result. Skipping the read wastes whole sessions re-deriving known results (e.g. re-adding a liquidity
  pre-filter Han never used; re-testing residual/nested-MH targets already rejected; chasing IC that the log shows is
  delisting-inflated). The log + papers are the accumulated memory of the sleeve.
- **Concretely:** if you are about to say "let's try X" or "why did Y happen", first `grep` the log for X/Y and check
  the REJECTED list and the mandatory papers. Cite what you found. Only then propose.
- **The papers ARE the priors.** Han (2022) defines the DM architecture (bimodality → reclassification → RET) and why
  no liquidity filter is needed; the Stop-Regressing/HL-Gauss papers define why CE-over-a-histogram beats MSE. If a
  proposal contradicts a mandatory paper's central result, say so and justify it before testing.

---

## PHASE 0 — SUBSTRATE (MANDATORY, no exceptions)

**Every test from now on loads data from `DATAHUB.py` and backtests through `DATAHUB` (daily) or `BACKTEST.py`
(monthly). NO external / ad-hoc data loaders.** No more `pd.read_parquet("tiingo_daily_close.parquet")` +
bespoke `me`/`elig`/`mdv` boilerplate in each script, and no `deep_momentum_xgb.load_broad_universe_tiingo`.

- **Data**: `from DATAHUB import DataHub; hub = DataHub()`. Daily-native, split-coherent mcap. It owns the ONE
  universe definition (`hub.elig("liquid"|"relaxed")`), the price/return/volume panels, fundamentals, and mcap.
  Monthly strategies (MOM/DM) resample from the same hub (`hub.me`, `hub.m_px`, `hub.mret`, `hub.synth`).
- **Backtest**: daily → `hub.backtest_daily(W, H, lag, cost_bps)` (honest lag=2, per-side bps). Monthly →
  `BACKTEST.backtest(...)` with `BACKTEST.tiered_transaction_costs` / `tiered_borrow_fees`.
- **Why**: one coherent frame = no silent incoherence (the split-basis mcap bug came from mixing adjusted price
  with as-reported shares across ad-hoc loaders). Comparable numbers require identical substrate.

---

## SCRIPT HYGIENE — ONE reusable harness per sleeve (MANDATORY, no exceptions)

**Each sleeve gets AT MOST 2 research scripts, which you OVERWRITE — never a new file per idea.**

| sleeve | harness (reuse + overwrite) | interactive | permanent log | papers |
|---|---|---|---|---|
| momentum | `mom_research.py` | `mom_research.ipynb` | `MOM_research.md` | `papers/mom/` |
| value/quality | `vq_research.py` | `vq_research.ipynb` | `VALUE_research.md` | `papers/vq/` |
| mean-reversion | `mr_research.py` | `mr_research.ipynb` | `MR_research.md` | `papers/mr/` |

**The loop (every idea):**
1. **Read** the sleeve's `_research.md` + its `⚠️ MANDATORY READING` papers (Phase R).
2. **Edit only the `EXPERIMENT` block** at the bottom of `<sleeve>_research.py` (or a notebook cell). The harness
   reuses `CURRENT BEST/<sleeve>_layer.py` for the leak-free pool + the standard scorers + the `net()` reporter, so
   a test is a few lines, not a new file.
3. **Run.** If the result is noteworthy → append a **T-entry to `<sleeve>_research.md`** (the permanent record).
4. **Overwrite** the `EXPERIMENT` block for the next idea. **DO NOT** create `research/<one_off>.py`.

**Rule:** the `.py`/`.ipynb` is disposable SCRATCH; the `_research.md` is the PERMANENT record. If a finding isn't in
the `.md`, it doesn't exist. (We let `research/` accumulate to 90+ one-off scripts before enforcing this — never again.)
Only the sleeve harnesses + shared data/audit utilities live in `research/`; one-offs are deleted after logging.

---

## NOTEWORTHY IDEAS (cross-sleeve idea bank — vetted-from-literature, applicable beyond one sleeve)

**1. The "curve" / PARAMETER-ENSEMBLE instead of single-point optimisation** (Quantitativo, "Trading the mean
reversion curve"; PJ Sutherland). *Don't optimise one threshold/horizon/target and trade it — that's fragile to
parameter instability + luck.* Instead build a SPECTRUM of the same strategy across a reasonable parameter range
(their example: RSI2 ∈ {5,10,15,20,25,30}, six books) and **dynamically allocate** — trailing-window (504d) max-Sharpe
weights, monthly rebalance — letting adaptive weighting discover the current regime. Their result: 25.7% ann / SR 1.14
vs Nasdaq-100 0.89; allocation shifted (RSI2=10 got 49% of the time, RSI2=25 only 18%).
- **Cross-sleeve use:** applies to ANY sleeve's tunable — MOM target (tval↔return, which we found is
  universe-dependent, [[target-universe-dependent]]) or horizon; VQ value/quality mix; MR rarity threshold. Ensemble
  the spectrum + adaptive weight rather than crowning one point.
- **CAVEAT (ours):** we already found STATIC horizon-*blending* is FLAT for MOM (T20/T21) — but that was equal-average,
  NOT adaptive regime-weighting; the adaptive version is a distinct, untested frontier. Also trailing-Sharpe weights can
  chase/overfit (needs the Phase-4 audits). Connects to [[adaptivity-and-frontier]] (online expert aggregation / BOA)
  and virtue-of-complexity (go-complex-then-shrink). Test as a CONSTRUCTION overlay, judged on net book SR, before adopting.

---

## THE WALKTHROUGH — run these 8 steps identically for EVERY idea

The point of a fixed walkthrough: **every idea gets identical treatment → results are directly comparable.**
Same universe, same baseline, same metric row, same audits, same ledger. No idea gets a bespoke test.

**STEP 1 — EXPLAIN, then write the idea CARD** (Phase 1 UNDERSTAND → Phase 2 DEFINE). Explain the idea in
plain words BEFORE formalizing; if you can't fill the top half, you don't understand it well enough to test:
```
IDEA (plain words): <what the thing IS — one paragraph a non-quant could follow>
MECHANISM/THEORY:   <WHY it should work — the economic/statistical reason, NOT "the model will learn it">
PRIOR ART:          <what funds/academics do here; the closest known result and its sign>
NAIVE BASELINE:     <the simplest thing that might ALREADY capture this — the arm the idea must beat>
EXPECTED SIGN/SIZE: <your prior on direction & rough magnitude, written before running>
──────────── formalize only after the above is clear ────────────
ID:            <short-slug>                 DATE: <yyyy-mm-dd>   TRIAL #: <running count>
HYPOTHESIS:    <X will change METRIC by ≥Δ because MECHANISM>
THE ONE CHANGE: <the single axis varied vs baseline>
BASELINE ARM:  <the exact control, run in the same script/pool — usually the NAIVE BASELINE above>
PRE-REG BAR:   <metric + Δ, written BEFORE running>          KILL IF: <condition>
```

**STEP 2 — Build on the STANDARD HARNESS.** Copy the canonical pool/walk; change ONLY the one axis.
Same features, construction (rank→decile→equal-weight→dollar-neutral→H=1), embargo=max-horizon. Baseline
arm in the SAME script. (This is what makes numbers comparable — never a fresh bespoke script.)

**STEP 3 — SCREEN** (top-1000, seeds=1). Emit the STANDARD METRIC ROW for the arm AND the baseline:
`| id | universe | gSR | nSR | ann | maxDD | turn | IC |`. Same columns, every time.

**STEP 4 — GATE 1 (screen).** `nSR(arm) − nSR(baseline)` vs the pre-reg bar.
→ below bar: **REJECT** (Step 8). → within 0.05 of bar: re-screen at seeds=3. → clears: continue.

**STEP 5 — AUDIT** (Phase 4 A–C). IC⇄Sharpe agreement · combo-pop check · robustness (Spearman/ex-outlier)
· correlation audit if combining (lead-lag/rolling) · name-overlap vs factor-corr. Any red flag → REJECT/NEEDS-MORE.

**STEP 6 — CONFIRM** (FULL universe, seeds=5). Re-run the survivor on the deployment substrate. Emit the
standard metric row again (now the row that counts).

**STEP 7 — GATE 2 (confirm).** Clears the bar NET, on full, at seeds=5, after audits? → **ADOPT**. Else → REJECT.

**STEP 8 — RECORD in the LEDGER.** Append the standard metric row + verdict to the results ledger below,
AND write rejections to memory with the number + reason. Every idea ends here, pass or fail.

### Results ledger (append-only — the comparability artifact)

Every idea's standard row lives in ONE table so they're read side-by-side. Keep it current.

| id | date | trial# | universe | seeds | gSR | nSR | ann | maxDD | turn | IC | vs base | verdict |
|----|------|--------|----------|-------|-----|-----|-----|-------|------|----|---------|---------|
| baseline-rankspace | 2026-07-09 | — | full | 5 | 1.63 | 1.06 | 18.8% | −35.3% | 2.14 | — | — | champion |
| dm-native-fulluniv | 2026-07-09 | — | full | 1 | — | 1.36 | 17.3% | −21.4% | — | +0.33 | NEEDS-MORE (seeds=5/CPCV) |
| _<next idea>_ | | | | | | | | | | | | |

---

## Phase 0 — SUBSTRATE (fixed; identical for every test)

The data and universe are a constant, not a knob. Fix them once so results are comparable across ideas.

- **Universe (canonical):** full liquid cross-section — `price > $5`, rolling-252 coverage > 0.9,
  monthly `$-volume > $5M`. NO top-N cap unless the idea is explicitly about capacity (then state it).
  Report `avg universe size (2011+)` on every run so universe drift is visible.
- **Returns:** `pct_change(fill_method=None)` (no fill across gaps); mask UPSIDE glitches `r ≥ +100%`;
  keep real losses incl. −100%; impute delisting `−30%` in the month a name stops trading.
- **OOS window:** evaluate 2011+ only (pre-2011 is the training warm-up / FFD-fit window).
- **Two universes allowed, but always labelled:** `full` (breadth/capacity truth) and `top-1000`
  (fast screening testbed). A screening result on top-1000 is DIRECTIONAL only — confirm the winner on
  `full` before believing the level. Never compare a top-1000 number to a full-universe number.

---

## Phase 1 — UNDERSTAND (explain the idea before any code)

Understanding is the gate, not a formality. Most bad tests are bad because the idea was never articulated — you
cannot design a fair control or read a result for an idea you can't explain. Before writing a line of backtest
code, write the idea out in plain words and answer, honestly:

1. **What is it, concretely?** Describe the mechanism a non-quant could follow. "A covariance-based gross dial
   that cuts exposure when markets couple" — not "an XGBoost on 25 features."
2. **Why should it work — theory, not hope.** Name the economic or statistical reason (a premium, a friction, a
   forecastable second moment). "The model will learn it" is NOT a mechanism. If the only argument is in-sample
   fit, stop here.
3. **What do funds/academics already do here?** Find the closest known result and its sign BEFORE testing. You
   are rarely first; the literature gives you the prior, the standard baseline, and the known failure modes
   (e.g. "factor timing is hard"; "vol targeting helps equities via the leverage effect").
4. **What naive baseline might ALREADY capture this?** The single most important question. Name the simplest
   thing (trailing vol, VIX, 12-1 momentum, equal-weight) that could produce the same result — that IS your
   control. An idea only earns its complexity by beating it end-to-end.
5. **Is the target even forecastable?** Second moments (vol/covariance) are forecastable; first moments
   (returns/timing) largely are not. An idea that needs return-timing to work is fighting the wall — say so up
   front and set expectations accordingly.
6. **State your prior:** expected sign and rough magnitude, in writing, before running. A result that violates a
   well-founded prior is more likely a bug than a discovery — you only notice if you wrote the prior down.

**The recurring lesson this phase exists to enforce: predictiveness ≠ portfolio value.** A better forecast
(higher IC) repeatedly fails to beat the naive baseline once it passes through construction and costs (jump-model
regimes, the XGBoost vol model, HAR/multi-horizon forecasts all lost this way). Understanding WHY an edge would
convert to net SR — through which consumer, against which baseline — is the whole job of Phase 1. Only when you
can explain all six do you proceed to DEFINE.

---

## Phase 2 — DEFINE (formalize the understood idea)

State it in writing. If you can't fill these in, the idea isn't ready to test.

1. **Hypothesis (falsifiable):** "Change X will improve METRIC by ≥ Δ, because MECHANISM."
2. **One change only.** Vary a single axis vs the baseline. If you must vary two, you cannot attribute
   the result — split into two tests. (Corollary: never trust a combo that pops when its parts are flat.)
3. **Baseline / control arm** is mandatory and run in the SAME script, SAME pool. The number that
   matters is `arm − baseline`, not the arm's absolute value.
4. **Pre-register the success bar** (the Δ and the metric) BEFORE seeing results. Prevents post-hoc
   goal-shifting. Default bars: net-SR Δ ≥ +0.10 that survives Phase 4, or IC Δ ≥ +0.005.
5. **Name the failure mode** you'd accept as killing it (e.g., "if net SR doesn't beat baseline, drop it").

---

## Phase 3 — MEASURE (the backtest — non-negotiable rules)

- **ONE ENGINE ONLY: `BACKTEST.py`.** Never hand-roll a backtest loop. A research script's job is to
  produce a **weights DataFrame** (dates×tickers) + the rebalance `signal_dates`, then call
  `BACKTEST.backtest(weights, prices, freq, lag, signal_dates, transaction_cost=tiered_transaction_costs(...),
  borrow_fee=tiered_borrow_fees(...))` and read the metrics dict. Ad-hoc `book()`/`costed()` loops are
  BANNED — they silently diverge on execution timing, cost/borrow accounting, drift, and short penalties
  (we shipped weeks of flat-bps, same-bar-execution reversion numbers this way before catching it).
  Rules that follow from using the engine: use `lag≥1` for realistic next-bar execution (lag=0 only for
  month-end signal→next-month, the monthly convention); pass **tiered** trade+borrow costs, never a flat
  scalar; if the model uses adjusted returns (delisting, winsorized ticks) feed the engine a **synthetic
  price grid** `(1+returns).cumprod()` so it drifts on the exact returns you trade.
- **LEAK-FREE (embargo).** Training months must satisfy `t + H ≤ k` where H = max label horizon,
  k = prediction month. Implement as `tr = [j for j in past if keys[j] <= k - H]`. This is the single
  most important rule — the overlapping-label leak inflated gross SR ~2–3× before we caught it. Verify
  with a lead-lag check when in doubt (correlation of a signal to its target should peak at lag 0).
- **NET-OF-COST is the headline.** Always report GROSS and NET (tiered transaction costs on turnover +
  tiered borrow fees on shorts). Gross alone is misleading; the deploy decision is net. Also report
  **2-way turnover/mo** — cost problems are turnover problems.
- **Standard metric set, every run:** `net SR`, `gross SR`, `ann`, `maxDD`, `turnover`, and `IC`
  (mean monthly corr of prediction vs realized t+1). Report all — they diverge and each catches a
  different failure.
- **Bagging discipline:** screen at `seeds=1` (direction only); bag the FINAL winner at `seeds=5`
  (≈ +0.09 variance-reduction). Never declare a champion at seeds=1.
- **Same pool, within-run comparison.** All arms share one pool build → apples-to-apples. Different
  scripts/samples are NOT comparable (different start dates, different embargo, different universe).
- **Construction held fixed** across arms unless construction IS the idea: rank on the score, top/bottom
  decile, equal-weight, dollar-neutral, hold H=1 (or state the holding rule).

---

## Phase 4 — ANALYZE (decision + audits — an idea is not "done" until it passes these)

Run EVERY audit relevant to the idea type. A raw Sharpe improvement is not a result until it survives.

**A. Consistency audits (always):**
- **IC ⇄ Sharpe agreement.** A Sharpe gain with FLAT IC is an overfit/construction artifact, not
  signal. Real signal moves both. (This is how we caught the kalman+facmom "combo pop.")
- **Robust-loss sanity:** if a change only helps via extreme months, check Spearman/ex-outlier.

**B. Overfitting signatures (reject if present):**
- Parts flat/negative but the COMBINATION pops → overfit interaction (52w+skew, branch-features,
  kalman+facmom all failed this way). Reject unless each part independently helps.
- Result appears only at one universe / one seed / one period → not robust.

**C. Correlation & combination audits (whenever combining or claiming diversification):**
- **Audit the correlation before trusting it:** lead-lag (peak must be at lag 0 = aligned), Pearson vs
  Spearman (agreement = not outlier-driven), ex-top-5-months, rolling-window stability.
- **Distinguish name-overlap from return-correlation.** Low name overlap + high return corr = COMMON
  FACTOR EXPOSURE, not the same bet (fix by neutralizing). High name overlap = literally the same book.
- **Combination level:** same-direction low-corr forecasts → SIGNAL level (average to denoise, raises
  IC). Opposite-direction premia → STRATEGY level (blend returns; signal-averaging would cancel alpha).
- **Weighting:** for a FEW similar-Sharpe streams use equal-weight / risk-parity. Do NOT use mean-
  variance / NCO (they overfit μ; proven worse here). MV/NCO only earn their keep with many streams.

**D. Verdict (one of three, in writing):**
- **ADOPT** — beats baseline past the pre-registered bar, survives A–C, confirmed on `full` + `seeds=5`.
- **REJECT** — record WHY in memory so it's never re-tested (see [[mom-sleeve-settled]]).
- **NEEDS-MORE** — promising but an audit is inconclusive; state the specific follow-up.

---

## Standing rules (learned the hard way)

- **The signal is where the money is, but NET is the constraint.** Optimize gross signal in screening;
  never call anything deployable without the net number + purge embargo.
- **Representation > architecture.** Target/feature transforms (rank space, horizon) beat model-class
  swaps. Test those first.
- **For a diversifying sleeve, standalone Sharpe is the WRONG objective** — orthogonality / combined
  Sharpe is. Judge sleeves on `combined`, not `standalone`.
- **Universe/feature diversity decorrelates more than horizon does** (DM-native same-universe corr 0.74
  vs external-DM 0.55). Diversify by changing the universe/features/premium, not the momentum horizon.
- **Kill fast.** seeds=1 top-1000 first; only promote survivors to full/seeds=5. Don't run heavy
  confirmations on ideas that failed the cheap screen.
- **Record every rejection in memory** with the number and the reason. The catalog is the asset.
- **PER-SLEEVE RESEARCH LOG (MOM / VALUE-QUALITY / MEAN-REVERSION only).** Each of the three sleeves keeps a
  living `research/<SLEEVE>_research.md` (`MOM_research.md`, `VALUE_research.md`, `MR_research.md`) with (a) the
  current WORKING THESIS at the top and (b) an append-only TEST LOG. **Every test run on that sleeve is appended
  immediately** in the format `ID · date · hypothesis · method (script, universe, seeds, metric) · result ·
  conclusion (ADOPT/REJECT/NEEDS-MORE)` — i.e. exactly how it was tested and what it concluded, so the reasoning
  is reproducible and the thesis is always current. Update the top-of-file thesis whenever a test changes it. This
  is mandatory for these three sleeves (it does NOT apply to infra/state/risk/execution work).
- **LIMIT NEW SCRIPTS — reuse, don't proliferate.** One coherent script per line of enquiry, not one per
  variant. Before creating a `research/*.py`, first look for an existing harness to extend (sweep via an
  arg/loop/env flag, add a column) — a target sweep, a feature-IC table, a leg decomposition are each ONE
  script that takes a list, not N near-duplicate files. Fold parameters (seeds, variant, universe) behind
  `os.environ`/argv rather than copy-pasting. Promote a proven result by editing the PRODUCTION sleeve
  (`MOM.py`/`VALUE.py`/…) in place, then delete or archive the scratch script. Rule of thumb: if two
  scripts share >70% of their body, they should have been one. Fewer, richer, parameterised scripts —
  the reviewer (and future you) can't hold 60 one-off files in their head.

---

## KNOWN WEAKNESSES of this protocol (and mitigations)

This pipeline is disciplined but not bulletproof. It is a **single-path frequentist backtest** framework;
these are its real failure modes. Read before trusting any "ADOPT".

1. **Multiple-testing / search overfitting (the BIGGEST hole).** Per-idea pre-registration does NOT
   control the family-wise error across the *many* ideas we test. Run 40 ideas against a "+0.10 SR" bar
   and ~2–3 pass by pure luck. → **Mitigation:** keep a running TRIAL COUNT; compute the **Deflated
   Sharpe Ratio** (López de Prado) that adjusts the bar for the number of trials; treat a lone winner
   among many tries with suspicion.
2. **Single walk-forward path → no confidence interval.** One embargoed walk gives ONE Sharpe, not a
   distribution — we can't say how likely it's real. → **Mitigation:** for any champion, run **CPCV**
   (combinatorial purged CV) to get many backtest paths, a Sharpe distribution, and the **Probability of
   Backtest Overfitting (PBO)**. One number is a point estimate, not evidence.
3. **Overfit baseline / champion selection bias.** Each arm is compared to a baseline that is ITSELF the
   survivor of prior testing — so "beats baseline" can be beating an already-lucky number. → **Mitigation:**
   also report improvement vs a NAIVE benchmark (e.g., raw 12-1 momentum), not only vs the champion;
   periodically re-validate the champion from scratch.
4. **"Fixed substrate" choices are themselves overfit.** Decile cutoff 10%, quarterly retrain, feature
   windows, the [t4,t5,t6] horizon, seeds=5 — all were *selected on this data* and then frozen. Treating
   them as constants hides that overfitting. → **Mitigation:** re-audit the frozen choices occasionally
   under nested CV; never present them as first-principles.
5. **Cost model is a single point estimate.** "Net" assumes the tiered cost model is truth; real impact
   is nonlinear and AUM-dependent. → **Mitigation:** report net at **1× and 2× costs**; an edge that dies
   at 2× is not deployable. Add a capacity/√-impact check before scaling.
6. **Sharpe/IC fixation ignores tails and regimes.** Sharpe is blind to skew/kurtosis; average correlation
   hides crisis co-crashes (momentum's killer). → **Mitigation:** gate champions on skew, kurtosis,
   worst-month, and **tail/crisis correlation**, not just mean Sharpe + maxDD. Correlations for weighting
   must be checked conditionally (do they spike in drawdowns?).
7. **seeds=1 screening → false negatives.** A real idea can fail the cheap screen by seed luck. →
   **Mitigation:** re-screen borderline rejects (within ~0.05 SR of the bar) at seeds=3 before killing.
8. **No forward / out-of-sample-DATASET validation.** Backtest ADOPT ≠ live-ready; and everything is on
   ONE dataset (Tiingo US). → **Mitigation:** paper-trade before deploy; validate the champion on a
   SECOND universe/market (the international-DM direction) — a different dataset is the strongest OOS test
   and, per this session, also the strongest DIVERSIFIER.
9. **"One change only" can miss real interactions.** Conservative by design (avoids overfit combos) but
   can under-explore genuine synergies. → **Mitigation:** when a combo is theory-motivated, test it AND
   its parts; adopt the combo only if the interaction survives Phase 4 (parts need not each win, but the
   combo must not be a lone pop with flat IC).

**Meta-weakness:** the protocol enforces *rigor per test* but not *skepticism about the whole program*.
The single most important habit it cannot encode: **prefer fewer, theory-driven tests over many
data-mined ones** — every extra trial raises the false-discovery rate. When in doubt, don't test; think.

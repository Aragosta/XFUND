# RESEARCH PROTOCOL — fool-proof pipeline for testing ideas

**Every new idea (feature, target, model, combination, execution, sleeve) passes through the same four
phases, in order, with no shortcuts.** The phases are MECE: each does exactly one job, together they
cover the whole pipeline.

```
Phase 0  SUBSTRATE     fixed universe + data hygiene        (set once, never per-idea)
Phase 1  DEFINE        one hypothesis, one change, a control, a pre-registered bar
Phase 2  MEASURE       leak-free · net-of-cost · standard metrics · bagging discipline
Phase 3  ANALYZE       decision criteria + overfitting/robustness audits → verdict
```

The mantra: **full universe → clearly DEFINE → clearly MEASURE → clearly ANALYZE.**

---

## THE WALKTHROUGH — run these 8 steps identically for EVERY idea

The point of a fixed walkthrough: **every idea gets identical treatment → results are directly comparable.**
Same universe, same baseline, same metric row, same audits, same ledger. No idea gets a bespoke test.

**STEP 1 — Write the idea CARD** (Phase 1). Fill the template exactly; if you can't, it's not ready:
```
ID:            <short-slug>                 DATE: <yyyy-mm-dd>   TRIAL #: <running count>
HYPOTHESIS:    <X will change METRIC by ≥Δ because MECHANISM>
THE ONE CHANGE: <the single axis varied vs baseline>
BASELINE ARM:  <the exact control, run in the same script/pool>
PRE-REG BAR:   <metric + Δ, written BEFORE running>          KILL IF: <condition>
```

**STEP 2 — Build on the STANDARD HARNESS.** Copy the canonical pool/walk; change ONLY the one axis.
Same features, construction (rank→decile→equal-weight→dollar-neutral→H=1), embargo=max-horizon. Baseline
arm in the SAME script. (This is what makes numbers comparable — never a fresh bespoke script.)

**STEP 3 — SCREEN** (top-1000, seeds=1). Emit the STANDARD METRIC ROW for the arm AND the baseline:
`| id | universe | gSR | nSR | ann | maxDD | turn | IC |`. Same columns, every time.

**STEP 4 — GATE 1 (screen).** `nSR(arm) − nSR(baseline)` vs the pre-reg bar.
→ below bar: **REJECT** (Step 8). → within 0.05 of bar: re-screen at seeds=3. → clears: continue.

**STEP 5 — AUDIT** (Phase 3 A–C). IC⇄Sharpe agreement · combo-pop check · robustness (Spearman/ex-outlier)
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

## Phase 1 — DEFINE (before writing any backtest code)

State it in writing. If you can't fill these in, the idea isn't ready to test.

1. **Hypothesis (falsifiable):** "Change X will improve METRIC by ≥ Δ, because MECHANISM."
2. **One change only.** Vary a single axis vs the baseline. If you must vary two, you cannot attribute
   the result — split into two tests. (Corollary: never trust a combo that pops when its parts are flat.)
3. **Baseline / control arm** is mandatory and run in the SAME script, SAME pool. The number that
   matters is `arm − baseline`, not the arm's absolute value.
4. **Pre-register the success bar** (the Δ and the metric) BEFORE seeing results. Prevents post-hoc
   goal-shifting. Default bars: net-SR Δ ≥ +0.10 that survives Phase 3, or IC Δ ≥ +0.005.
5. **Name the failure mode** you'd accept as killing it (e.g., "if net SR doesn't beat baseline, drop it").

---

## Phase 2 — MEASURE (the backtest — non-negotiable rules)

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

## Phase 3 — ANALYZE (decision + audits — an idea is not "done" until it passes these)

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
   its parts; adopt the combo only if the interaction survives Phase 3 (parts need not each win, but the
   combo must not be a lone pop with flat IC).

**Meta-weakness:** the protocol enforces *rigor per test* but not *skepticism about the whole program*.
The single most important habit it cannot encode: **prefer fewer, theory-driven tests over many
data-mined ones** — every extra trial raises the false-discovery rate. When in doubt, don't test; think.

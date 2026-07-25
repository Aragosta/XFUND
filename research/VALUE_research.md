# VALUE_research — working thesis & test log (VALUE-QUALITY sleeve)

**Living document.** Working thesis at the top; every test appended to the LOG with its EXACT method + conclusion.
Same discipline as [RESEARCH_PROTOCOL](RESEARCH_PROTOCOL.md). Sibling logs: `MOM_research.md`, `MR_research.md`.
Created 2026-07-21 when the sleeve was consolidated into `CURRENT BEST/vq_layer.py` (superseding the scattered
VALUE.py / QUALITY.py / FUNDQ.py). See memory [[feature-eng-dm-mr-vq]].

---

## WORKING THESIS (current)

The sleeve = **QARP (quality at a reasonable price)**: an XGBoost multi-output regressor predicts each name's
**forward profitability** (GP/A + ROE @ +12mo, sector-neutral Gaussian rank — a genuinely forecastable fundamental,
R²~0.3-0.57 vs ~0.009 for returns, [[target-snr-return-vs-secondmoment]]); the book trades **cheap × predicted-to-be-
quality**, sector-neutral decile L/S, beta-neutralized. Full fundamental feature panel (value / quality / safety /
growth / Piotroski / net-issuance), no price → orthogonal to momentum *by input*, but NOT by output (see T2).

**Honest status:** the sleeve WORKS standalone (net SR ~0.58) but its role is unresolved. The reason value/quality
is usually run is DIVERSIFICATION (the textbook value↔momentum −0.5). **That negative correlation does NOT exist on
our data/period (T2) — value is +0.18 to +0.52 correlated with momentum in liquid-US 2015-2026.** So VQ is currently
a momentum-ADJACENT standalone book, not a diversifier. Open question: is there ANY VQ construction that is
orthogonal-or-negative to MOM/DM here, or is value simply not the diversifier in this universe (→ lean on MR + 13F)?

## ⚠️ MANDATORY READING (Phase R) — read before proposing/testing anything on this sleeve
1. **Asness, Frazzini, Pedersen (2019), "Quality Minus Junk"** — QMJ = profitability + growth + **safety** + payout;
   quality is defensive. (Our sleeve has the profitability + safety legs; growth/payout partial.)
2. **Asness, Moskowitz, Pedersen (2013), "Value and Momentum Everywhere"** — the canonical value↔momentum −0.5 and
   why the 50/50 combo dominates. ⚠️ **CAVEAT (T2): this is a 1927-2013 / broad-universe average; it does NOT
   reproduce in our liquid-US 2015-2026 sample.** Do not assume the −corr — measure it.
3. **Novy-Marx (2013), "The Other Side of Value: Gross Profitability"** — GP/A is the cleanest quality signal;
   fundamental momentum drives price momentum (why a "no-price" fundamental book still co-moves with MOM).
4. **Eisfeldt-Kim-Papanikolaou (2020) / Arnott et al., "Reports of Value's Death…"** — intangibles-adjusted book
   (capitalize R&D) as the modern value fix. ⚠️ CAVEAT (T2): on our data it makes value MORE momentum-like.

---

## TEST LOG (append-only)

**T1 · 2026-07-21 · CONSOLIDATION + QARP standalone.** Built `vq_layer.py` (clean class API like mr/mom_layer):
forward-profitability target, 25 fundamental feats incl. intangibles-adjusted book (`rnd` enabled in DATAHUB RAW),
Piotroski F, QMJ safety leg, op-yield-to-EV, net-issuance. Method: seeds=1, liquid tier, sector-neutral decile L/S,
beta-neut, lag=1, tiered cost+borrow, OOS 2015-09+. **Result: net SR 0.58 (QARP), maxDD −30.9%; value-winter drag
2013-19 then strong 2021-25.** Conclusion: **ADOPT as the consolidated sleeve** (supersedes VALUE/QUALITY/FUNDQ);
the engine + feature panel are the keeper.

**T2 · 2026-07-21 · Is VQ a momentum DIVERSIFIER? (the reason to run value at all).** Correlated QARP + raw value +
12-1 momentum returns (2015-2026, built from scratch). **Result: corr(QARP,MOM)=+0.52; corr(RAW VALUE,MOM)=+0.18.**
Neutralization audit (does stripping sector/beta hide a −corr?): sector+beta-neut +0.18 / RAW-no-neut **+0.32** /
sector-neut-only +0.18 — removing neutralization makes it MORE positive → NOT a construction artifact. **Conclusion:
the classic value↔momentum −0.5 is ABSENT here; value & momentum are positively correlated in liquid-US 2015-2026.**
Mechanism: QARP buys quality-growth = momentum winners; even raw value (buys the fallen) is only +0.18 because 2022
had value AND momentum both work. **⇒ dumb value is DOMINATED (SR −0.41, no −corr); QARP is the better construction;
value/quality is NOT the diversifier in this universe.** REJECT the "keep a dumb-value leg for the −corr" idea.

**T3 · MOM cross-reference (from mom_research, same session).** Best MOM long-only: multi-horizon return [t4,t5,t6]
rank-space net SR 0.27 > single-6mo 0.24 > tval@6 0.16 (seeds=1, full univ). tval has highest IC (0.102) but lowest
SR (IC≠portfolio value). Not a verdict (seeds=1 screen); logged for the MOM sleeve.

## OPEN / NEXT (not yet tested)
- Is ANY VQ variant orthogonal-or-negative to MOM/DM here? (deep-value only / long-horizon reversal value /
  value-spread-timed). If none → accept VQ is momentum-adjacent and lean diversification on MR + 13F.
- Judge VQ on the COMBINED book (does QARP lift the MOM+DM+MR ERC despite +0.52 corr?), not standalone.
- QMJ growth + payout legs; value-trap knockout via the profitability model as a FILTER (not ranker).

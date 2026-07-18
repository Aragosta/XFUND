# CURRENT BEST

The **settled, production-grade version of each layer** — one file per layer, only the version that survived
testing. Research/experiments live in `../research/`; this folder holds the winners.

Architecture:  `DATA → STATE → sleeves (MOM, DM, …) → ERC → gross dial → META → execution`

Rule for this folder: a layer goes here only when its design is decided by **end-to-end evidence** (SR / Sortino /
maxDD vs a naive baseline), not by in-sample fit or forecast IC. If a fancier version doesn't beat the simple one
on the book, the simple one stays.

---

## `state_layer.py` — the STATE layer  ✅ settled

**One object: a covariance forecast `Σ_t`** of daily factor + sector returns (EWMA, half-life 63d = the
RiskMetrics standard). Everything the layer outputs is a read-out of `Σ_t`.

**What it produces (all leak-free):**
| output | what it is | who consumes it |
|---|---|---|
| `gross` | vol-target gross dial in `[0.4, 1.0]` = `clip(median(σ_port)/σ_port, 0.4, 1.0)` | risk layer (exposure multiplier) |
| `sigma_port` | forecast monthly book vol `√(wᵀΣw)` | risk layer (`.book_gross(weights)` for the real book) |
| `Sigma` | 4×4 factor covariance per month | ERC / construction |
| `enb` | Meucci effective-number-of-bets (modern absorption) | **diagnostic only** |
| `surprise` | Mahalanobis distance (modern turbulence) | **diagnostic only** |
| `feat` | lean regime features (macro cols, dispersion, value spread, sentiment) | sleeves |

**Why it's this and nothing more** — each rejected on our own data (see `../research/`, memory `state-layer`):
- The **dial adds value**: static + coupling/vol-target overlay cuts maxDD ~−19% → ~−11%, lifts Sortino, at
  trivial turnover. It's the one place the state layer beats a naive baseline end-to-end.
- **No hand-picked constants**: the old `1 − 0.6·pct` / `[0.4,1.0]` overlay was replaced by a vol-target off `Σ`;
  the only number left is an *estimated* trailing median (not fit to P&L) + a `0.4` leverage guardrail.
- **Rejected, proven not to add end-to-end value:** HMM / GMM / statistical **jump-model** regimes (lose to the
  one-line dial); a **multi-output XGBoost** vol forecaster (IC ~0.5 but loses to trailing vol); **mode-return
  targets** squeeze/reversal/value/mom-crash (return-timing = the wall, AQR "factor timing is hard"); **dynamic /
  multi-horizon** forecasts HAR / ensemble / +VIX (more *accurate*, but the coarse dial can't spend the accuracy).
- **The recurring lesson:** predictiveness (IC) ≠ portfolio value. The binding constraint is the dial's
  resolution, not the forecast — so the *only* open lever is a finer **daily no-trade-band dial** (would also make
  the forward-looking VIX-blend pay). Not yet built; would be the next thing to promote here if it tests out.

**Run / use:**
```bash
python3 "CURRENT BEST/state_layer.py"        # builds, saves to ./out/, prints diagnostics
```
```python
from state_layer import StateLayer
st = StateLayer().build()
dial = st.book_gross(my_post_erc_factor_weights)   # book-specific vol-target gross
```
Outputs are written to `CURRENT BEST/out/` (`state_cov`, `state_dial`, `state_feat` parquets).

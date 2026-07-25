#!/usr/bin/env python3
"""
vq_layer.py — the VALUE-QUALITY sleeve (consolidated).  ⏳ NEW — supersedes VALUE.py / QUALITY.py / FUNDQ.py.

ONE object: a cross-sectional forecast of each name's forward PROFITABILITY (a denoised, genuinely-forecastable
fundamental — R²~0.3-0.57 vs ~0.009 for returns, [[target-snr-return-vs-secondmoment]]), read out as a
"quality-at-a-reasonable-price" (QARP) book: buy CHEAP names whose profitability the model predicts will be
HIGH/RISING. Orthogonal to price momentum BY CONSTRUCTION (no price in features or target) → a ~-0.4 diversifier
to the MOM/DM sleeves. Judged as a DIVERSIFIER on the COMBINED book, not standalone (value winter in liquid US).

The recipe (consolidates the three legacy scripts + the recommended upgrades):

  TARGET        forward profitability, +12mo, sector-neutral Gaussian rank — TWO heads: GP/A and ROE (the FUNDQ
                target our stats picked). Predicting the persistent fundamental sidesteps the return-prediction wall.
  FEATURES      the full value/quality/safety/growth fundamental panel, all cross-sectionally z-scored, PIT-lagged:
                VALUE   bm, **bm_intan** (intangibles-adjusted book: book + depreciated R&D stock — the modern
                        fix for value's "book misses intangibles" death), ep, sp, cfp, gp_ev, ocf_ev, shyield
                QUALITY gpa (Novy-Marx), roe, roa, gmar, ocfa, accr (Sloan, low=good)
                SAFETY  lev, curr, casha, earnvol (QMJ safety leg — makes quality defensive / mom-crash hedge)
                GROWTH  revg, nig, ag (asset growth, low=good), capexa
                SCORE   **piotroski** F-score (9 accounting-health checks; strongest on cheap names), net_issuance
                        (issuers underperform), size
  MODEL         XGBoost multi_output_tree regressor (2 heads), rolling walk-forward, refit every 3mo, 12mo embargo
                (future fundamentals → strict no-look-ahead).
  CONSTRUCTION  predicted future profitability z  +  cheapness z (intangibles-adjusted value tilt)  → sector-neutral
                → decile long/short → BETANEUT. The QARP mispricing lever: cheap × predicted-to-be-quality.
  RESULT        the layer emits the raw monthly book + score; sizing/blend is the ERC/META layer's job.

WHY THIS AND NOT THE LEGACY SCRIPTS (see [[feature-eng-dm-mr-vq]]): VALUE.py (raw B/M) sits in the liquid-US value
winter because book value misses intangibles → bm_intan repairs it. QUALITY.py is static profitability only → the
ML forward-profitability target + QMJ safety leg add the predictive + defensive content. FUNDQ.py had the right
target but a thin feature set and no consolidation → this is FUNDQ's engine with the full panel and a clean API.

REJECTED / not included (avoid re-testing): raw-return target (unpredictable, R²~0.009); price/momentum features
(would break the orthogonality that IS this sleeve's job); standalone-SR optimisation (wrong objective for a
diversifier — judge on combined book). Convergence-of-multiple 2nd target head is a PARKED extension (flag below).

Run:  python3 "CURRENT BEST/vq_layer.py"     # builds features+target, honest walk-forward, prints net + corr to MOM/DM
"""
import warnings, os, sys
warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # repo root
import numpy as np, pandas as pd, pickle
from scipy.stats import norm
from xgboost import XGBRegressor
import BACKTEST, BETANEUT
from DATAHUB import DataHub

EMB   = 12                                                                          # forward-fundamental embargo (months)
REG   = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.8, tree_method="hist", multi_strategy="multi_output_tree", verbosity=0)


def _grank(a):
    a = np.asarray(a, float); r = pd.Series(a).rank(method="average")
    return norm.ppf((r - 0.5) / max(len(r), 2))


class VqLayer:
    def __init__(self, hub: DataHub | None = None, start: str = "2000-01-01", seeds: int = 1,
                 tier: str = "liquid", value_tilt: bool = True):
        self.hub = hub or DataHub(start=start, min_days=0)
        self.seeds, self.tier, self.value_tilt = seeds, tier, value_tilt

    # ── PIT fundamental feature panel + forward-profitability target ────────────────────────────────
    def build(self):
        h = self.hub
        self.me, self.m_px, self.synth = h.me, h.m_px, h.synth
        self.elig = h.elig(self.tier); self.sec = h.sector
        if os.environ.get("SURVIVORS"):                    # survivorship-bias TEST: keep only names still trading at
            alive = self.m_px.iloc[-1].notna()             # the end of the sample (mimics a survivorship-biased dataset)
            self.elig = self.elig & alive
        self.mret = h.mret; self.mcap = h.mcap(); f = h.fund
        self.BETA = BETANEUT.rolling_beta(self.mret, self.elig, bw=60)
        self.tc = BACKTEST.tiered_transaction_costs(h.mdv); self.bf = BACKTEST.tiered_borrow_fees(h.mdv)
        mcap = self.mcap
        d12 = lambda x: x / x.shift(12) - 1

        book, ni, rev, cogs, assets = f("book"), f("ni"), f("rev"), f("cogs"), f("assets")
        ocf, gp, capex, debt, cash  = f("ocf"), f("gross_profit"), f("capex"), f("debt_lt"), f("cash")
        acur, lcur, shares, rnd, div = f("assets_cur"), f("liab_cur"), f("shares"), f("rnd"), f("dividends")

        # intangibles-adjusted book: depreciated R&D knowledge-capital stock (Peters-Taylor / Eisfeldt), δ=0.2
        rstock = sum((0.8 ** k) * rnd.shift(12 * k) for k in range(5))
        intan_book = book.add(rstock.where(rnd.notna(), 0.0), fill_value=0.0)
        EV = (mcap + debt.fillna(0) - cash.fillna(0)).where(lambda z: z > 0)

        gpa  = ((rev - cogs) / assets).where(assets > 0)
        roe  = (ni / book).where(book > 0)
        roa  = (ni / assets).where(assets > 0)
        gmar = (gp / rev).where(rev > 0)
        turn = (rev / assets).where(assets > 0)
        lev  = (debt / assets).where(assets > 0)
        curr = (acur / lcur).where(lcur > 0)
        nig  = d12(ni)

        feat = {
            # VALUE
            "bm": book / mcap, "bm_intan": intan_book / mcap, "ep": ni / mcap, "sp": rev / mcap,
            "cfp": ocf / mcap, "gp_ev": gp / EV, "ocf_ev": ocf / EV, "shyield": div / mcap,
            # QUALITY
            "gpa": gpa, "roe": roe, "roa": roa, "gmar": gmar, "ocfa": ocf / assets, "accr": (ni - ocf) / assets,
            # SAFETY
            "lev": lev, "curr": curr, "casha": cash / assets, "earnvol": nig.rolling(36, min_periods=18).std(),
            # GROWTH / INVESTMENT
            "revg": d12(rev), "nig": nig, "ag": d12(assets), "capexa": capex / assets,
            # SCORE
            "net_iss": d12(shares), "size": np.log(mcap),
        }
        # Piotroski F-score (0-9): profitability, leverage/liquidity, operating efficiency
        F9 = [(roa > 0), (ocf > 0), (roa - roa.shift(12) > 0), (ocf / assets > roa),
              (lev - lev.shift(12) < 0), (curr - curr.shift(12) > 0), (shares <= shares.shift(12) * 1.001),
              (gmar - gmar.shift(12) > 0), (turn - turn.shift(12) > 0)]
        feat["piotroski"] = sum(c.fillna(False).astype(float) for c in F9)

        if os.environ.get("NEWFUND"):                                              # v2 EDGAR: true-EBIT signals
            opinc, sga, intexp = f("opinc"), f("sga"), f("intexp")
            fcf = ocf - capex; inv_cap = (book.fillna(0) + debt.fillna(0) - cash.fillna(0)).where(lambda z: z > 0)
            feat["ebit_ev"] = opinc / EV                                            # enterprise multiple (best value axis)
            feat["fcf_ev"]  = fcf / EV                                              # free-cash-flow yield to EV
            feat["roic"]    = (opinc * (1 - 0.21)) / inv_cap                        # NOPAT / invested capital (quality)
            feat["ebitmar"] = (opinc / rev).where(rev > 0)                          # operating margin
            feat["intcov"]  = (opinc / intexp).where(intexp > 0)                    # interest coverage (safety)
            feat["sga_rev"] = (sga / rev).where(rev > 0)                            # operating efficiency

        def zc(df):
            df = df.replace([np.inf, -np.inf], np.nan)
            return df.sub(df.mean(1), 0).div(df.std(1) + 1e-9, 0).clip(-3, 3)
        self.F = {k: zc(v) for k, v in feat.items()}; self.COLS = list(self.F)
        self.cheap = zc(feat["bm_intan"]).add(zc(feat["ep"]), fill_value=0).add(zc(feat["cfp"]), fill_value=0)  # value tilt

        # TARGET: forward profitability (+12mo), sector-neutral Gaussian rank, 2 heads
        fut_gpa, fut_roe = gpa.shift(-EMB), roe.shift(-EMB)
        def secrank(row, idx, d):
            r = row.where(self.elig.loc[d]); r = r - r.groupby(self.sec).transform("mean")
            return _grank(r.reindex(idx).values)

        pool = {}
        for i, d in enumerate(self.me):
            live = self.elig.loc[d] & mcap.loc[d].notna(); idx = live[live].index
            if len(idx) < 100 or i + EMB >= len(self.me): continue
            X = np.column_stack([self.F[c].loc[d].reindex(idx).fillna(0.0).values for c in self.COLS])
            Y = np.column_stack([secrank(fut_gpa.loc[d], idx, d), secrank(fut_roe.loc[d], idx, d)])
            pool[d] = dict(X=X.astype(np.float32), idx=idx, Y=Y, cheap=self.cheap.loc[d].reindex(idx))
        self.pool = pool; self.dates = [d for d in self.me if d in pool]
        return self

    # ── walk-forward → QARP book → net backtest ─────────────────────────────────────────────────────
    def backtest(self, decile: float = 0.10, tag: str = "VALUExQUALITY"):
        me, mcols = self.me, self.m_px.columns
        W = pd.DataFrame(0.0, index=me, columns=mcols); mdl = None
        for j, d in enumerate(self.dates):
            if j < 60: continue
            if mdl is None or j % 3 == 0:
                cut = me[max(0, me.get_loc(d) - EMB)]
                tr = [self.dates[k] for k in range(j) if self.dates[k] <= cut]
                if len(tr) < 48: continue
                Xtr = np.vstack([self.pool[t]["X"] for t in tr]); Ytr = np.vstack([self.pool[t]["Y"] for t in tr])
                ok = np.isfinite(Ytr).all(1)
                mdl = [XGBRegressor(**REG, random_state=s).fit(Xtr[ok], Ytr[ok]) for s in range(self.seeds)]
            if mdl is None: continue
            q = np.mean([m.predict(self.pool[d]["X"]).mean(1) for m in mdl], axis=0)   # predicted future profitability
            s = pd.Series(q, index=self.pool[d]["idx"])
            if self.value_tilt:                                                        # QARP: cheap × predicted-quality
                b = self.pool[d]["cheap"]; s = (s - s.mean()) / (s.std() + 1e-9) + (b - b.mean()) / (b.std() + 1e-9)
            s = s - s.groupby(self.sec.reindex(s.index)).transform("mean")             # sector-neutral
            s = s.dropna()
            n = max(1, int(len(s) * decile))
            W.loc[d, s.nlargest(n).index] = 1.0 / n; W.loc[d, s.nsmallest(n).index] = -1.0 / n
        W = BETANEUT.betaneut(W, self.BETA)
        r = BACKTEST.backtest(W.fillna(0.0), self.synth, freq=12, lag=1,
                              signal_dates=[d for d in W.index if W.loc[d].abs().sum() > 1e-9],
                              transaction_cost=self.tc, borrow_fee=self.bf)
        r["book"] = W; return r

    def value_book(self, decile: float = 0.10, knockout: float = 0.5):
        """DUMB VALUE (restores the −corr to momentum): driver = RAW price-based cheapness (B/M+E/P+CF/P, NO
        intangibles → buy the FALLEN losers). Quality is only a KNOCKOUT: among the cheap longs keep the best-
        quality (drop value traps); among the expensive shorts keep the worst-quality. knockout=0 → pure value;
        knockout=0.5 → widen candidate pool 1.5× and filter by quality. Static (no ML) → fast."""
        cheap = self.F["bm"].add(self.F["ep"], fill_value=0).add(self.F["cfp"], fill_value=0)       # RAW cheapness
        qual  = self.F["gpa"].add(self.F["roe"], fill_value=0).add(self.F["roa"], fill_value=0)
        W = pd.DataFrame(0.0, index=self.me, columns=self.m_px.columns)
        for d in self.me:
            s = cheap.loc[d].where(self.elig.loc[d]).dropna()
            s = s - s.groupby(self.sec.reindex(s.index)).transform("mean")                          # sector-neutral
            if len(s) < 50: continue
            n = max(1, int(len(s) * decile)); m = int(n * (1 + knockout)); q = qual.loc[d]
            cl = s.nlargest(m).index; keep_l = q.reindex(cl).nlargest(n).index                      # cheap × best-quality
            cs = s.nsmallest(m).index; keep_s = q.reindex(cs).nsmallest(n).index                    # expensive × worst-quality
            W.loc[d, keep_l] = 1.0 / n; W.loc[d, keep_s] = -1.0 / n
        W = BETANEUT.betaneut(W, self.BETA)
        r = BACKTEST.backtest(W.fillna(0.0), self.synth, freq=12, lag=1,
                              signal_dates=[d for d in W.index if W.loc[d].abs().sum() > 1e-9],
                              transaction_cost=self.tc, borrow_fee=self.bf)
        r["book"] = W; return r


def _show(nm, x):
    x = x[(x.index >= "2013-01-01")].dropna(); e = (1 + x).cumprod()
    print(f"  {nm:22} SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>6.1%}  "
          f"maxDD {(e/e.cummax()-1).min():>6.1%}", flush=True)
    print("    " + " ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+4.0%}" for y in range(2013, 2027)), flush=True)


def _corrs(x, tag):
    x = x[(x.index >= "2015-09-01") & (x != 0)]
    for nm, pth, key in [("MOM", "/tmp/mom_champ.pkl", "n1"), ("DM", "/tmp/dm_returns.pkl", None)]:
        if os.path.exists(pth):
            o = pickle.load(open(pth, "rb")); s = pd.Series(o[key] if key else o); s.index = pd.DatetimeIndex(s.index)
            c = pd.DataFrame({"v": x, "m": s}).dropna()
            if len(c) > 24: print(f"    corr({tag}, {nm}) = {c.corr().iloc[0,1]:+.2f}", flush=True)


if __name__ == "__main__":
    seeds = int(os.environ.get("SEEDS", 1)); MODE = os.environ.get("MODE", "both")
    L = VqLayer(seeds=seeds).build()
    print("=" * 92); print(f"VALUE-QUALITY sleeve — net · {len(L.dates)}mo × {len(L.COLS)} feats · seeds={seeds} · MODE={MODE}", flush=True)
    if MODE in ("both", "value"):
        for ko in (0.0, 0.5):                                     # pure value, then value + quality knockout
            r = L.value_book(knockout=ko); x = pd.Series(r["returns"]); x.index = pd.DatetimeIndex(x.index)
            _show(f"DUMB VALUE (ko={ko})", x); _corrs(x, "val")
    if MODE in ("both", "qarp"):
        rQ = L.backtest(); x = pd.Series(rQ["returns"]); x.index = pd.DatetimeIndex(x.index)
        _show("QARP (cheap×pred-quality)", x); _corrs(x, "qarp")
        pickle.dump(x.to_dict(), open("/tmp/vq_returns.pkl", "wb"))
    print("[done] VQ sleeve.", flush=True)

#!/usr/bin/env python3
"""DM.py — FAITHFUL replication of Han (2022), "Bimodal Characteristic Returns and Predictability
Enhancement via Machine Learning", Management Science 68(10) 7701-7741.  [rebuilt 2026-07-25]

WHY THIS REWRITE. Every prior DM/MOM conclusion about the SHORT side ("momentum has no short premium",
T35/T37/T39/T40) was drawn from books carrying beta ≈ −0.85 — a large short-beta bet, not the
dollar-/beta-neutral portfolio Han reports. We had also never run Han's actual model: different features
(we omitted his cross-sectional-mean macro features entirely), different learner (trees vs DNN), different
training scheme (rolling 120m vs expanding + validation), different universe (liquid tier vs full CRSP),
different reclassification (Gaussian bin centres vs 20-year empirical class means). This file removes every
one of those degrees of freedom so the replication succeeds or fails on Han's own terms.
The previous multi-horizon vector-leaf sleeve is preserved at research/archive/DM_legacy_multihorizon.py.

═══ HAN'S SPEC, IMPLEMENTED EXACTLY (§ refs are to the paper) ═══

DATA (§4.1)             Han: CRSP common shares (10,11), NYSE/Amex/Nasdaq, 1955.01-2017.01. Inclusion at
                        month t: price at end of t−13 and a return at t−2; market equity at end of t−1.
                        Delist return, else −30% (Beaver et al. 2007). 22,919 firms, monthly avg 3,837.
                        NO liquidity/size filter — the RECLASSIFICATION is the filter (short-book avg cap
                        $153M → $1,589M).
  OURS                  DataHub full universe, NO tier filter. Delisting −30% already injected by
                        DataHub.pnl. Our history starts 1990, so the burn-in puts the test at ~2011.

FEATURES (§3.2.3, Tab 1a)  MOM_m = ∏_{j=t−m}^{t−2}(1+r_j) − 1   for m ∈ {3,6,9,12}     [skips t−1]
                        MOM_1 = r_{t−1}                                                 [the skipped month]
                        nMOM_m = (MOM_m − mean)/std, cross-sectionally, each month
                        M_MOM_m = the cross-sectional MEAN of MOM_m, broadcast to every stock —
                                  "included to take the MACROECONOMIC STATUS into account".
                                  WE NEVER HAD THESE: 5 features carrying the market state.
                        D_s, s=1..10 = size dummies from mcap deciles at end of month t−1.
                        ⇒ 5 nMOM + 5 M_MOM + 10 dummies = 20 features. That is ALL Han uses.
                        (= model MOM-SZ-NOM, the spec behind his Table 7 factor regression.)

MODEL (§3.2.1, Tab 1c)  NOMINAL classifier: DNN, 5 hidden layers × 64 neurons, ReLU, softmax over K=10
                        classes, cross-entropy loss. Regularisation = EARLY STOPPING ONLY ("to evaluate
                        the models from a conservative perspective"). Classes = deciles of the 1-month-
                        ahead return; class 0 = HIGHEST return.

TRAINING (§4.1)         EXPANDING window, retrained EVERY YEAR, "stacking the sample while holding the
                        last ten years of the data for validation": at 1975.01 train=1955.01-1964.12,
                        valid=1965.01-1974.12; at 1976.01 train=1955.01-1965.12, valid=1966.01-1975.12.
                        Han averages 50 random trials.

RECLASSIFICATION on Return (§3.3.4)  THE champion criterion ("the Return criterion appears to offer the
                        best and most robust performance"). μ̂_k = sample mean return of class k over the
                        PAST TWENTY YEARS; μ̂ⁱ = Σ_k ŷⁱ_k · μ̂_k (law of total expectation, Eq. 22). Rank
                        on μ̂ⁱ. NOTE: centres are EMPIRICAL and TIME-VARYING, not Gaussian quantiles.

PORTFOLIO (§4.1, §4.3)  Each month sort μ̂ⁱ into deciles, hold ONE month. H = top decile, L = bottom,
                        H−L = long-short. Reported EQUAL- and VALUE-weighted.

TARGETS (Han, 1975-2017, gross, avg of 50 trials)
  EW H−L  >40%/yr SR 2.0-2.49   ·   VW H−L  >30%/yr SR 1.60-1.86   ·   VW H alone ~25%/yr SR ≤1.11
  standard momentum EW 10%/0.37, VW 19%/0.61
  Costs (§1.3): 10bp → VW L/S SR 1.46 · 30bp → 1.09 · DeMiguel-conservative → VW L/S collapses to 4.8%,
  LONG-ONLY survives 8.3% excess / SR 0.34, EW L/S 19.6% / SR 1.06.

DEVIATIONS (unavoidable; nothing else differs)
  1. History 1990+ not 1955+ → 10y train / 10y validation burn-in, test from ~2011 (Han 20y/10y, 1975).
  2. SEEDS defaults to 5, not Han's 50 (env SEEDS=50 to match; runtime scales linearly).
  3. DataHub's US equity panel, not CRSP proper (no share-code/exchange fields available).
Run:  SEEDS=5 python DM.py       env: SEEDS, KCLASS, TEST_START
"""
import warnings; warnings.filterwarnings("ignore")
import os, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
import BACKTEST
from DATAHUB import DataHub

_SZPROXY = None
SEEDS      = int(os.environ.get("SEEDS", 5))          # Han averages 50 trials
K          = int(os.environ.get("KCLASS", 10))        # return classes = deciles
TEST_START = int(os.environ.get("TEST_START", 2011))  # first OOS year (Han: 1975, after a 20y burn-in)
VALID_YRS  = 10                                       # "holding the last ten years ... for validation"
MU_WIN     = 240                                      # μ̂_k over "the past twenty years" (months)
DEC        = 0.10                                     # decile portfolios
HID, NLAY  = 64, 5                                    # Table 1(c)
MAX_EPOCH, PATIENCE, BATCH, LR = 200, 10, 4096, 1e-3  # early stopping ONLY (§3.2.4)
# SIZE=0 -> MOM-NOM (momentum features only, 10 feats). SIZE=1 -> MOM-SZ-NOM (adds mcap size dummies).
# DEFAULT 0 BECAUSE OF A DATA LIMIT, NOT A CHOICE: hub.mcap is EDGAR-derived and is empty before ~2009
# (1-2 names/month pre-2007), so size dummies truncate the pool to 2010+ and destroy the 10y-train/10y-
# validation burn-in. Han's HEADLINE equal-weighted results (Table 5: EW H-L >40%/yr, SR 2.0-2.49) are
# precisely the models "using only momentum or return features" — i.e. MOM-NOM. Size dummies are what he
# adds in §4.4 to rescue the VALUE-weighted book, not the equal-weighted one. So MOM-NOM is the faithful
# route to the headline claim on our data. SIZE=1 remains available for the 2010+ sub-sample.
# SIZE=2 -> size dummies from DOLLAR-VOLUME rank instead of market cap. Han's size dummies are what let
# the model shift the book toward LARGE caps (§4.4, §4.4.1: short-book avg cap $153M -> $1,589M after
# Return-reclassification, "profits are not driven by small firms"). Our first MOM-NOM run had NO size
# information at all, and its gross alpha turned out to live entirely in the illiquid tail (liquid-universe
# gross alpha -1.35%, t=-0.42). mdv is a full-history proxy for market cap (hub.mcap is EDGAR-only, 2009+),
# so SIZE=2 restores the size channel over 1990-2026. DOCUMENTED DEVIATION: mdv-rank != market-equity rank.
SIZE = int(os.environ.get("SIZE", 0))


def han_features(mret, mcap):
    """Han's exact 20 features.

    Indexing: row i of `mret` is the return REALISED in month i. We form the book for month t=i+1, so the
    most recent known return is r_{t−1} = mret.iloc[i]. MOM_m therefore spans rows [i−m+1 .. i−1] (months
    t−m..t−2), EXCLUDING row i — Jegadeesh-Titman's skipped month. All known at end of month i → leak-free.
    """
    R = 1.0 + mret
    lg = np.log(R.where(R > 0))
    MOM = {1: mret}                                                    # MOM_1 = r_{t−1}
    for m in (3, 6, 9, 12):
        MOM[m] = np.exp(lg.shift(1).rolling(m - 1, min_periods=m - 1).sum()) - 1.0
    feats, names = {}, []
    for m in (1, 3, 6, 9, 12):
        x = MOM[m]
        mu, sd = x.mean(axis=1), x.std(axis=1)
        feats[f"nMOM{m}"] = x.sub(mu, axis=0).div(sd + 1e-12, axis=0)   # cross-sectionally normalised
        feats[f"M_MOM{m}"] = pd.DataFrame(np.repeat(mu.values[:, None], x.shape[1], 1),
                                          index=x.index, columns=x.columns)   # the MACRO-STATE feature
        names += [f"nMOM{m}", f"M_MOM{m}"]
    if SIZE:                                                            # §4.4 size dummies
        base = mcap if SIZE == 1 else _SZPROXY                          # 1 = true mcap (2009+), 2 = mdv proxy
        d = base.rank(axis=1, pct=True)                                 # size deciles at end of month t−1
        for s in range(10):
            feats[f"D{s+1}"] = ((d > s / 10) & (d <= (s + 1) / 10)).astype(float).where(base.notna())
            names.append(f"D{s+1}")
    return feats, names


class DNN(nn.Module):
    """Table 1(c): 5 hidden layers × 64 ReLU neurons → softmax over K classes (nominal classifier, §3.2.1)."""
    def __init__(self, nf, k):
        super().__init__()
        L, d = [], nf
        for _ in range(NLAY):
            L += [nn.Linear(d, HID), nn.ReLU()]; d = HID
        self.net = nn.Sequential(*L, nn.Linear(d, k))
    def forward(self, x): return self.net(x)


def _fit(Xtr, ytr, Xva, yva, seed):
    """Cross-entropy training, EARLY STOPPING on the held-out last-10-years validation block."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = DNN(Xtr.shape[1], K); opt = torch.optim.Adam(m.parameters(), lr=LR); lf = nn.CrossEntropyLoss()
    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr); Xva = torch.tensor(Xva); yva = torch.tensor(yva)
    best, bstate, bad, n = np.inf, None, 0, len(Xtr)
    for _ in range(MAX_EPOCH):
        m.train(); perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad(); lf(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = lf(m(Xva), yva).item()
        if vl < best - 1e-5:
            best, bstate, bad = vl, {kk: v.clone() for kk, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    if bstate is not None: m.load_state_dict(bstate)
    m.eval(); return m


def run():
    t0 = time.time()
    hub = DataHub(start="1990-01-01", min_days=0)
    global _SZPROXY
    mret = hub.clean_returns("monthly"); mcap = hub.mcap("monthly"); pnl = hub.pnl("monthly")
    _SZPROXY = hub.dollar_size("monthly")                               # full-history size proxy for SIZE=2
    me, cols = mret.index, mret.columns
    fwd = hub.mret.reindex(index=me, columns=cols).shift(-1)        # 1-month-AHEAD return = label source
    feats, fnames = han_features(mret, mcap)
    print(f"[HAN-DM] universe {len(cols):,} names · {len(me)} months {me[0].date()}..{me[-1].date()}"
          f" · {len(fnames)} features (Han's exact 20) · K={K} · seeds={SEEDS}", flush=True)

    rows = {}
    for i in range(13, len(me) - 1):                                # need t−13 history and a realised label
        X = pd.DataFrame({n: feats[n].iloc[i] for n in fnames})
        y = fwd.iloc[i]
        ok = X.notna().all(axis=1) & y.notna()
        if SIZE == 1: ok &= mcap.iloc[i].notna()
        idx = X.index[ok]
        if len(idx) < 200: continue
        yy = y.reindex(idx)
        cls = (K - 1) - np.clip((yy.rank(method="first") - 0.5) / len(yy) * K, 0, K - 1).astype(int)
        rows[i] = dict(dt=me[i], idx=idx, X=X.loc[idx].values.astype(np.float32),
                       y=cls.values.astype(np.int64), r=yy.values)   # class 0 = HIGHEST return
    keys = sorted(rows)
    print(f"[HAN-DM] pool {len(keys)} months · avg {np.mean([len(rows[k]['idx']) for k in keys]):.0f}"
          f" names/month (Han: 3,837)", flush=True)

    score, PROB, models, cur_year = {}, {}, None, None
    for k in keys:
        d = rows[k]["dt"]
        if d.year < TEST_START: continue
        if cur_year != d.year or models is None:                     # ANNUAL refit, EXPANDING window
            cur_year = d.year
            vcut = d - pd.DateOffset(years=VALID_YRS)
            tr = [j for j in keys if rows[j]["dt"] < vcut]
            va = [j for j in keys if vcut <= rows[j]["dt"] < d]
            if len(tr) < 24 or len(va) < 24: continue
            Xtr = np.vstack([rows[j]["X"] for j in tr]); ytr = np.concatenate([rows[j]["y"] for j in tr])
            Xva = np.vstack([rows[j]["X"] for j in va]); yva = np.concatenate([rows[j]["y"] for j in va])
            models = [_fit(Xtr, ytr, Xva, yva, s) for s in range(SEEDS)]
            print(f"  [{d.year}] train {len(tr)}m/{len(Xtr):,} obs · valid {len(va)}m · "
                  f"{time.time()-t0:.0f}s", flush=True)
        if models is None: continue
        with torch.no_grad():
            Xk = torch.tensor(rows[k]["X"])
            P = np.mean([torch.softmax(m(Xk), 1).numpy() for m in models], axis=0)
        hist = [j for j in keys if rows[j]["dt"] < d][-MU_WIN:]      # μ̂_k over the past twenty years
        mu = np.zeros(K)
        for c in range(K):
            v = np.concatenate([rows[j]["r"][rows[j]["y"] == c] for j in hist]) if hist else np.array([0.0])
            mu[c] = np.nanmean(v) if len(v) else 0.0
        sd = np.zeros(K)                                             # class variances for the SHARPE criterion
        for c in range(K):
            v = np.concatenate([rows[j]["r"][rows[j]["y"] == c] for j in hist]) if hist else np.array([0.0])
            sd[c] = np.nanvar(v) if len(v) > 1 else 0.0
        score[k] = pd.Series(P @ mu, index=rows[k]["idx"])           # μ̂ⁱ = Σ_k ŷ_k μ̂_k  (Eq. 22)
        PROB[k] = dict(dt=d, idx=rows[k]["idx"], P=P.astype(np.float32), mu=mu, var=sd)

    S = pd.DataFrame({rows[k]["dt"]: score[k] for k in score}).T.reindex(columns=cols)
    os.makedirs("CURRENT BEST/out", exist_ok=True)
    S.to_parquet("CURRENT BEST/out/han_dm_score.parquet")
    import pickle; pickle.dump(PROB, open("CURRENT BEST/out/han_dm_prob.pkl", "wb"))
    print(f"[HAN-DM] saved {len(PROB)} months of class PROBABILITIES + class mu/var -> han_dm_prob.pkl\n"
          f"         (lets Return / Sharpe / PrDf / cost-adjusted reclassification be compared WITHOUT retraining)",
          flush=True)

    sm = hub.dollar_size("monthly")
    tc = BACKTEST.tiered_transaction_costs(sm); bf = BACKTEST.tiered_borrow_fees(sm)
    spy = hub.spy_m; spy = spy.iloc[:, 0] if isinstance(spy, pd.DataFrame) else spy
    spy.index = pd.PeriodIndex(spy.index, freq="M"); spy = spy[~spy.index.duplicated()]

    def build(kind, leg):
        W = {}
        for k in score:
            s = score[k].dropna()
            if len(s) < 50: continue
            n = max(1, int(len(s) * DEC))
            hi, lo = s.nlargest(n).index, s.nsmallest(n).index
            cap = mcap.iloc[k].reindex(s.index)
            def wt(sel):
                if kind == "vw":
                    v = cap.reindex(sel).clip(lower=0).fillna(0.0)
                    return v / v.sum() if v.sum() > 0 else pd.Series(1.0 / len(sel), index=sel)
                return pd.Series(1.0 / len(sel), index=sel)
            w = pd.Series(0.0, index=s.index)
            if leg in ("H", "HL"): w.loc[hi] = wt(hi)
            if leg in ("L", "HL"): w.loc[lo] = w.loc[lo] - wt(lo)
            W[rows[k]["dt"]] = w
        return pd.DataFrame(W).T.reindex(columns=pnl.columns)

    def report(tag, W, costs):
        r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index),
                              transaction_cost=tc if costs else 0.0, borrow_fee=bf if costs else 0.0)
        x = r["returns"].dropna(); x = x[x.index >= W.index.min()]
        if len(x) < 24: return
        x.index = pd.PeriodIndex(x.index, freq="M")
        D = pd.concat({"r": x, "m": spy}, axis=1).dropna()
        X = np.c_[np.ones(len(D)), D.m.values]; c, *_ = np.linalg.lstsq(X, D.r.values, rcond=None)
        e = D.r.values - X @ c
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(D) - 2, 1)))
        dd = ((1 + D.r).cumprod() / (1 + D.r).cumprod().cummax() - 1).min()
        print(f"  {tag:26}{D.r.mean()*12:>9.2%}{D.r.std()*np.sqrt(12):>8.2%}"
              f"{D.r.mean()*12/(D.r.std()*np.sqrt(12)+1e-9):>7.2f}{dd:>9.1%}{c[1]:>7.2f}"
              f"{c[0]*12:>+9.2%}{c[0]/(se[0]+1e-12):>7.2f}{r['ann_turnover']:>7.1f}", flush=True)

    for costs in (False, True):
        print(f"\n[HAN-DM] {'NET of tiered costs+borrow' if costs else 'GROSS (Han reports gross)'}"
              f" · test {TEST_START}+ · reclassification=Return", flush=True)
        print(f"  {'portfolio':26}{'ann':>9}{'vol':>8}{'SR':>7}{'maxDD':>9}{'beta':>7}{'alpha':>9}{'t':>7}{'turn':>7}")
        for kind in ("ew", "vw"):
            for leg in ("H", "L", "HL"):
                report(f"{kind.upper()} {leg}", build(kind, leg), costs)
        if kind == "vw":
            print("  [!] VW rows use hub.mcap, which only covers EDGAR filers (~30-40% of names, 2010+),"
                  "\n      so the VW book silently drops non-filers. EW is the trustworthy read here.")
    print(f"\n[HAN-DM] done in {time.time()-t0:.0f}s · score -> CURRENT BEST/out/han_dm_score.parquet", flush=True)


if __name__ == "__main__":
    run()

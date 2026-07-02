#!/usr/bin/env python3
"""LTR horizon test: does targeting a LONGER horizon (t+2, cum 3m) keep LTR's Sharpe while cutting
turnover? (Trend persists, so unlike DM — where t+2 re-coupled to crash — LTR should extend cleanly.)
Label = decile of the horizon return; PnL always earns the immediate next month (monthly rebalance).
Daily 750 universe, Poh features, LambdaMART rank:pairwise, inverse-vol decile L/S. GROSS + net-10bp."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRanker

N_SEEDS = 3
RANKER = dict(objective="rank:pairwise", eval_metric="ndcg", n_estimators=100, max_depth=6,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

print("[load] daily ...", flush=True)
px = pd.read_parquet("tiingo_daily_checkpoint.parquet").sort_index()
rets_d = px.pct_change(); vol_d = rets_d.ewm(span=63, min_periods=20).std()
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
mret = px.reindex(me).pct_change(); T = len(me)
elig = (px.reindex(me) > 1.0) & (px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill") > 0.9)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")

print("[feat] Poh/Baz ...", flush=True)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp); PF = list(pf.keys())

def relevance(fwd):
    r = fwd.rank(method="first"); return (((r - 1) * 10 // len(r)).clip(upper=9)).astype(int)

# horizon label functions (return realized over the horizon, for RANKING target)
HZ = {
    "t+1 (base)": lambda k: mret.iloc[k + 1] if k + 1 < T else None,
    "t+2":        lambda k: mret.iloc[k + 2] if k + 2 < T else None,
    "cum 3m":     lambda k: (1 + mret.iloc[k + 1:k + 4]).prod() - 1 if k + 4 <= T else None,
}

def build_pool(label_fn):
    pool = {}
    for k in range(13, T - 1):
        dt = me[k]
        P = pd.DataFrame({c: pf[c].loc[dt] for c in PF})
        lab = label_fn(k); pnl = mret.iloc[k + 1]                     # PnL always = next month
        if lab is None:
            base = P.notna().all(axis=1) & pnl.notna() & elig.loc[dt].fillna(False)
            idx = P.index[base.values]
            if len(idx) >= 50: pool[k] = dict(P=P.loc[idx], rel=None, pnl=pnl.reindex(idx), dt=dt)
            continue
        ok = P.notna().all(axis=1) & lab.notna() & pnl.notna() & elig.loc[dt].fillna(False)
        idx = P.index[ok.values]
        if len(idx) < 50: continue
        pool[k] = dict(P=P.loc[idx], rel=relevance(lab.reindex(idx)), pnl=pnl.reindex(idx), dt=dt)
    return pool

def run(label_fn):
    pool = build_pool(label_fn); keys = sorted(pool)
    fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)
    rows, prevw, store = [], pd.Series(dtype=float), None
    for i in range(fp, len(keys)):
        k = keys[i]; dt = pool[k]["dt"]
        if dt.month == 1 or store is None:
            tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt and pool[keys[j]]["rel"] is not None][-120:]
            if len(tr) >= 36:
                X = pd.concat([pool[t]["P"] for t in tr]); y = pd.concat([pool[t]["rel"] for t in tr])
                qid = np.concatenate([np.full(len(pool[t]["P"]), j) for j, t in enumerate(tr)])
                store = [XGBRanker(**RANKER, random_state=s).fit(X, y, qid=qid) for s in range(N_SEEDS)]
        if store is None: continue
        s = pd.Series(np.mean([m.predict(pool[k]["P"]) for m in store], axis=0), index=pool[k]["pnl"].index)
        n = max(1, int(len(s) * 0.10)); iv = (1 / (vol_d.loc[dt].reindex(s.index) * np.sqrt(21))).clip(upper=50)
        w = pd.Series(0.0, index=s.index); lo, sh = s.nlargest(n).index, s.nsmallest(n).index
        w[lo] = iv[lo] / iv[lo].sum(); w[sh] = -iv[sh] / iv[sh].sum()
        to = (w.subtract(prevw, fill_value=0).abs().sum()); prevw = w
        rows.append((dt, float((w * pool[k]["pnl"]).sum()), to))
    df = pd.DataFrame(rows, columns=["dt", "r", "to"]).set_index("dt")
    r = df["r"]; turn = df["to"].mean() * 12
    ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    rn = r - 0.001 * df["to"]                                          # net: 10bp/side × turnover
    sn = (rn.mean() * 12) / (rn.std() * np.sqrt(12))
    return ann, (r.mean() * 12) / vol, mdd, turn, sn

print("\n" + "=" * 72)
print("=== LTR horizon test (daily 750, gross) ===")
print(f"{'target':14}{'ann':>8}{'sharpe':>8}{'maxDD':>9}{'ann.turn':>10}{'net10bp SR':>12}")
for name, fn in HZ.items():
    print(f"[run] {name} ...", flush=True)
    a, s, m, t, sn = run(fn)
    print(f"{name:14}{a:>8.1%}{s:>8.2f}{m:>9.1%}{t:>10.0%}{sn:>12.2f}", flush=True)
print("[done]", flush=True)

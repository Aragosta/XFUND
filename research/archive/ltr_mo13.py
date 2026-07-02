#!/usr/bin/env python3
"""LTR MULTI-OUTPUT: train a t+1 ranker AND a t+3 ranker (same features), combine into one book
score = z(rank_t+1) + z(rank_t+3). Reports t+1, t+3, and COMBINED, plus corr between the two score
streams (if highly correlated, the combine adds little). Daily 750, inverse-vol decile L/S, gross."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRanker

N_SEEDS = 3
RANKER = dict(objective="rank:pairwise", eval_metric="ndcg", n_estimators=100, max_depth=6,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

px = pd.read_parquet("tiingo_daily_checkpoint.parquet").sort_index()
rets_d = px.pct_change(); vol_d = rets_d.ewm(span=63, min_periods=20).std()
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
mret = px.reindex(me).pct_change(); T = len(me)
elig = (px.reindex(me) > 1.0) & (px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill") > 0.9)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")

print("[feat] ...", flush=True)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp); PF = list(pf.keys())
def rel(fwd): r = fwd.rank(method="first"); return (((r - 1) * 10 // len(r)).clip(upper=9)).astype(int)

print("[pool] ...", flush=True)
pool = {}
for k in range(13, T - 3):
    dt = me[k]; P = pd.DataFrame({c: pf[c].loc[dt] for c in PF})
    l1, l2, pnl = mret.iloc[k + 1], mret.iloc[k + 3], mret.iloc[k + 1]
    ok = P.notna().all(axis=1) & l1.notna() & l2.notna() & pnl.notna() & elig.loc[dt].fillna(False)
    idx = P.index[ok.values]
    if len(idx) < 50: continue
    pool[k] = dict(P=P.loc[idx], r1=rel(l1.reindex(idx)), r2=rel(l2.reindex(idx)), pnl=pnl.reindex(idx), dt=dt)
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)

def z(a): a = pd.Series(a); return (a - a.mean()) / (a.std() + 1e-9)
print("[fit] t+1 & t+3 rankers ...", flush=True)
s1, s2 = {}, {}; st1 = st2 = None
for i in range(fp, len(keys)):
    k = keys[i]; dt = pool[k]["dt"]
    if dt.month == 1 or st1 is None:
        tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt][-120:]
        if len(tr) >= 36:
            X = pd.concat([pool[t]["P"] for t in tr]); qid = np.concatenate([np.full(len(pool[t]["P"]), j) for j, t in enumerate(tr)])
            y1 = pd.concat([pool[t]["r1"] for t in tr]); y2 = pd.concat([pool[t]["r2"] for t in tr])
            st1 = [XGBRanker(**RANKER, random_state=s).fit(X, y1, qid=qid) for s in range(N_SEEDS)]
            st2 = [XGBRanker(**RANKER, random_state=s).fit(X, y2, qid=qid) for s in range(N_SEEDS)]
            print(f"  [{dt.year}]", flush=True)
    if st1 is None: continue
    s1[k] = np.mean([m.predict(pool[k]["P"]) for m in st1], axis=0)
    s2[k] = np.mean([m.predict(pool[k]["P"]) for m in st2], axis=0)

def book(scorer):
    rows, prevw, sc_corr = [], pd.Series(dtype=float), []
    for k in s1:
        s = pd.Series(scorer(k), index=pool[k]["pnl"].index)
        n = max(1, int(len(s) * 0.10)); iv = (1 / (vol_d.loc[pool[k]["dt"]].reindex(s.index) * np.sqrt(21))).clip(upper=50)
        w = pd.Series(0.0, index=s.index); lo, sh = s.nlargest(n).index, s.nsmallest(n).index
        w[lo] = iv[lo] / iv[lo].sum(); w[sh] = -iv[sh] / iv[sh].sum()
        rows.append((pool[k]["dt"], float((w * pool[k]["pnl"]).sum()), w.subtract(prevw, fill_value=0).abs().sum())); prevw = w
    df = pd.DataFrame(rows, columns=["dt", "r", "to"]).set_index("dt"); r = df["r"]
    ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return r, ann, (r.mean() * 12) / vol, mdd, df["to"].mean() * 12

sc_corr = np.mean([np.corrcoef(s1[k], s2[k])[0, 1] for k in s1])
r1, a1, sh1, m1, t1 = book(lambda k: s1[k])
r2, a2, sh2, m2, t2 = book(lambda k: s2[k])
rc, ac, shc, mc, tc = book(lambda k: z(s1[k]).values + z(s2[k]).values)

print("\n" + "=" * 66)
print("=== LTR multi-output: t+1, t+3, COMBINED (daily 750, gross) ===")
print(f"{'signal':22}{'ann':>8}{'sharpe':>8}{'maxDD':>9}{'ann.turn':>10}")
for nm, a, sh, m, t in [("t+1", a1, sh1, m1, t1), ("t+3", a2, sh2, m2, t2), ("COMBINED z(t+1)+z(t+3)", ac, shc, mc, tc)]:
    print(f"{nm:22}{a:>8.1%}{sh:>8.2f}{m:>9.1%}{t:>10.0%}")
print(f"\navg cross-sectional corr(t+1 score, t+3 score) = {sc_corr:.2f}")
print(f"corr(t+1 returns, t+3 returns) = {pd.DataFrame({'a':r1,'b':r2}).dropna().corr().iloc[0,1]:.2f}")
print("[done]", flush=True)

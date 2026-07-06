#!/usr/bin/env python3
"""Volume/liquidity features for the MOM regressor — the first features ORTHOGONAL to price momentum.
Base = Poh+FFD (current sleeve). Add, on identical rows:
  amihud  : illiquidity = mean(|ret| / $vol) over 252d      (illiquid names / price-impact)
  dvtrend : log(mean $vol 63d / mean $vol 252d)             (liquidity growth / accumulation)
  abnvol  : mean $vol 21d / mean $vol 252d                  (recent volume shock)
Multi-output regressor, expanding+quarterly, clean broad top-1000, N_SEEDS=1, gross."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRegressor
import deep_momentum_xgb as d
from features import MOM_WINDOWS

N_SEEDS, TOPN, MINPX, MINDV, DELIST = 1, 1000, 5.0, 5e6, -0.30
REG = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex_like(px)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
m_px = px.reindex(me)
mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0)
V = m_px.notna().values; last = len(me) - 1
for j in range(V.shape[1]):
    w = np.where(V[:, j])[0]
    if len(w) and w[-1] < last - 1 and w[-1] + 1 < len(me): mret.iat[w[-1] + 1, j] = DELIST
dret = px.pct_change(fill_method=None); vol_d = dret.ewm(span=63, min_periods=20).std(); T = len(me)
dvd = px * vb                                                          # daily dollar volume
dvm = dvd.resample("ME").sum(); dvm.index = dvm.index.to_period("M")
mom_dvol = dvm.reindex(pd.PeriodIndex(me, freq="M")); mom_dvol.index = me
cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
eligm = (m_px > MINPX) & (cov > 0.9) & (mom_dvol > MINDV) & (mom_dvol.rank(axis=1, ascending=False) <= TOPN)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")
def zc(s): return (s - s.mean()) / (s.std() + 1e-9)
def zrow(df): return df.sub(df.mean(1), 0).div(df.std(1) + 1e-9, 0)

print("[feat] Poh + FFD ...", flush=True)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1).where(lambda z: z.abs() < 5); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp)
f2011 = int(np.argmax(me.year >= 2011)); ffd = d._ffd_from_training_window(m_px, f2011)
for m, v in ffd.items():
    z_ = v.reindex(me); pf[f"ffd{m}"] = z_.sub(z_.mean(1), 0).div(z_.std(1) + 1e-9, 0)
BASE_COLS = list(pf.keys())

print("[feat] volume ...", flush=True)
dv63 = dvd.rolling(63, min_periods=40).mean(); dv252 = dvd.rolling(252, min_periods=150).mean()
pf["amihud"]  = zrow(at_me((dret.abs() / dvd.replace(0, np.nan)).rolling(252, min_periods=120).mean()))
pf["dvtrend"] = zrow(at_me(np.log(dv63 / dv252)))
pf["abnvol"]  = zrow(at_me(dvd.rolling(21, min_periods=15).mean() / dv252))
VOL_ALL = ["amihud", "dvtrend", "abnvol"]; PF = list(pf.keys())

print("[pool] ...", flush=True)
pool = {}
for k in range(13, T - 3):
    dt = me[k]; el = eligm.loc[dt].fillna(False); idx0 = el.index[el.values]
    P = pd.DataFrame({c: pf[c].loc[dt].reindex(idx0) for c in PF}); pn = mret.iloc[k + 1].reindex(idx0)
    ok = P[BASE_COLS].notna().all(axis=1) & mret.iloc[k+1].reindex(idx0).notna() & mret.iloc[k+2].reindex(idx0).notna() & mret.iloc[k+3].reindex(idx0).notna()
    idx = P.index[ok.values]
    if len(idx) < 50: continue
    Y = np.column_stack([zc(mret.iloc[k+h].reindex(idx)).values for h in (1, 2, 3)])
    pool[k] = dict(P=P.loc[idx].fillna(0.0), Y=Y, pnl=pn.reindex(idx), dt=dt)
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)

def run(cols):
    rows, prevw, ms = [], pd.Series(dtype=float), None
    for i in range(fp, len(keys)):
        k = keys[i]; dt = pool[k]["dt"]
        if dt.month in (1, 4, 7, 10) or ms is None:
            tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt]
            if len(tr) >= 36:
                X = pd.concat([pool[t]["P"][cols] for t in tr]).values; Y = np.vstack([pool[t]["Y"] for t in tr])
                ms = [XGBRegressor(**REG, multi_strategy="multi_output_tree", random_state=s).fit(X, Y) for s in range(N_SEEDS)]
        if ms is None: continue
        sc = np.mean([m.predict(pool[k]["P"][cols].values).mean(1) for m in ms], axis=0)
        s = pd.Series(sc, index=pool[k]["pnl"].index); n = max(1, int(len(s) * 0.10))
        iv = (1 / (vol_d.loc[dt].reindex(s.index) * np.sqrt(21))).clip(upper=50)
        w = pd.Series(0.0, index=s.index); lo, sh = s.nlargest(n).index, s.nsmallest(n).index
        w[lo] = iv[lo] / iv[lo].sum(); w[sh] = -iv[sh] / iv[sh].sum()
        rows.append((dt, float((w * pool[k]["pnl"]).sum(skipna=True)), w.subtract(prevw, fill_value=0).abs().sum())); prevw = w
    df = pd.DataFrame(rows, columns=["dt", "r", "to"]).set_index("dt"); r = df["r"]
    ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return ann, (r.mean() * 12) / vol, mdd, df["to"].mean() * 12

print("\n" + "=" * 62)
print(f"=== MOM volume features (broad top-{TOPN}, gross) ===")
print(f"{'feature set':30}{'ann':>8}{'sharpe':>8}{'maxDD':>9}{'ann.turn':>10}")
for nm, cols in [("Poh+FFD (base)", BASE_COLS), ("+ amihud (illiq)", BASE_COLS + ["amihud"]),
                 ("+ dvtrend+abnvol (dynamics)", BASE_COLS + ["dvtrend", "abnvol"]),
                 ("+ all volume", BASE_COLS + VOL_ALL)]:
    print(f"[run] {nm} ...", flush=True); a, s, m, t = run(cols)
    print(f"{nm:30}{a:>8.1%}{s:>8.2f}{m:>9.1%}{t:>10.0%}", flush=True)
print("[done]", flush=True)

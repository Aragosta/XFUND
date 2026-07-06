#!/usr/bin/env python3
"""Is the MOM regressor a real anomaly? Factor-spanning regression (FF5 + MOM + STR, Newey-West HAC
t-stats) on the MOM sleeve AND the DM sleeve, side by side. Key questions:
  - MOM: significant alpha BEYOND the momentum factor? (β_MOM≈1 expected; is there α on top?)
  - DM vs MOM factor profiles: does DM load on STR/reversal and MOM on the MOM factor?
    (that would explain the ~0.17 correlation as FACTOR-LEVEL, not luck).
MOM stream recomputed (base Poh+FFD, top-1000, regressor, expanding+quarterly, N_SEEDS=1, gross);
DM stream loaded from /tmp/dm_streams.pkl (gross)."""
import warnings; warnings.filterwarnings("ignore")
import io, zipfile, pickle, requests
import numpy as np, pandas as pd, statsmodels.api as sm
from xgboost import XGBRegressor
import deep_momentum_xgb as d
from features import MOM_WINDOWS

# ── FF factors ──
def ff(url, cols):
    z = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=60).content))
    txt = z.read(z.namelist()[0]).decode("latin-1").splitlines(); rows = []
    for ln in txt:
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 2 and p[0].isdigit() and len(p[0]) == 6:
            try: rows.append([p[0]] + [float(x) for x in p[1:len(cols)+1]])
            except: pass
    df = pd.DataFrame(rows).set_index(0); df.columns = cols
    df.index = pd.to_datetime(df.index, format="%Y%m").to_period("M")
    return df.astype(float) / 100.0
print("[ff] downloading factors ...", flush=True)
f5 = ff("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
        ["MKT", "SMB", "HML", "RMW", "CMA", "RF"])
mom = ff("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip", ["MOM"])
strv = ff("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_CSV.zip", ["STR"])
FAC = pd.concat([f5[["MKT", "SMB", "HML", "RMW", "CMA"]], mom, strv], axis=1).dropna()
FCOLS = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "STR"]

# ── MOM stream (base Poh+FFD, top-1000, regressor) ──
print("[MOM] recompute base stream ...", flush=True)
px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex_like(px)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0)
Vv = m_px.notna().values; lastp = len(me) - 1
for j in range(Vv.shape[1]):
    w = np.where(Vv[:, j])[0]
    if len(w) and w[-1] < lastp - 1 and w[-1] + 1 < len(me): mret.iat[w[-1] + 1, j] = -0.30
vol_d = px.pct_change(fill_method=None).ewm(span=63, min_periods=20).std(); T = len(me)
dvm = (px * vb).resample("ME").sum(); dvm.index = dvm.index.to_period("M")
mdv = dvm.reindex(pd.PeriodIndex(me, freq="M")); mdv.index = me
cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6) & (mdv.rank(axis=1, ascending=False) <= 1000)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")
def zc(s): return (s - s.mean()) / (s.std() + 1e-9)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1).where(lambda z: z.abs() < 5); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp); BC = list(pf.keys())
ffd = d._ffd_from_training_window(m_px, int(np.argmax(me.year >= 2011)))
for m, v in ffd.items():
    z_ = v.reindex(me); pf[f"ffd{m}"] = z_.sub(z_.mean(1), 0).div(z_.std(1) + 1e-9, 0)
PF = list(pf.keys())
pool = {}
for k in range(13, T - 3):
    dt = me[k]; el = elig.loc[dt].fillna(False); idx0 = el.index[el.values]
    P = pd.DataFrame({c: pf[c].loc[dt].reindex(idx0) for c in PF}); pn = mret.iloc[k + 1].reindex(idx0)
    ok = P[BC].notna().all(axis=1) & mret.iloc[k+1].reindex(idx0).notna() & mret.iloc[k+2].reindex(idx0).notna() & mret.iloc[k+3].reindex(idx0).notna()
    idx = P.index[ok.values]
    if len(idx) < 50: continue
    Y = np.column_stack([zc(mret.iloc[k+h].reindex(idx)).values for h in (1, 2, 3)])
    pool[k] = dict(P=P.loc[idx].fillna(0.0), Y=Y, pnl=pn.reindex(idx), dt=dt)
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)
rows, prevw, ms = [], pd.Series(dtype=float), None
for i in range(fp, len(keys)):
    k = keys[i]; dt = pool[k]["dt"]
    if dt.month in (1, 4, 7, 10) or ms is None:
        tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt]
        if len(tr) >= 36:
            X = pd.concat([pool[t]["P"] for t in tr]).values; Y = np.vstack([pool[t]["Y"] for t in tr])
            ms = [XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0, multi_strategy="multi_output_tree", random_state=0)]
            ms[0].fit(X, Y)
    if ms is None: continue
    sc = ms[0].predict(pool[k]["P"].values).mean(1)
    s = pd.Series(sc, index=pool[k]["pnl"].index); n = max(1, int(len(s) * 0.10))
    iv = (1 / (vol_d.loc[dt].reindex(s.index) * np.sqrt(21))).clip(upper=50)
    w = pd.Series(0.0, index=s.index); lo, sh = s.nlargest(n).index, s.nsmallest(n).index
    w[lo] = iv[lo] / iv[lo].sum(); w[sh] = -iv[sh] / iv[sh].sum()
    rows.append((dt, float((w * pool[k]["pnl"]).sum(skipna=True))))
mom_stream = pd.Series(dict(rows))

# ── DM stream from cache ──
dm_g, _ = pickle.load(open("/tmp/dm_streams.pkl", "rb"))

def spanning(ret, name):
    r = ret.copy(); r.index = r.index.to_period("M")
    df = pd.concat([r.rename("y"), FAC], axis=1).dropna()
    res = sm.OLS(df["y"], sm.add_constant(df[FCOLS])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    a = res.params["const"] * 12; at = res.tvalues["const"]
    b = {c: (res.params[c], res.tvalues[c]) for c in FCOLS}
    print(f"\n{name}  ({len(df)} months)")
    print(f"  alpha = {a:>6.1%}/yr  (t = {at:>5.2f})   R2 = {res.rsquared:.2f}")
    print("  " + "  ".join(f"{c}:{b[c][0]:>5.2f}(t{b[c][1]:>4.1f})" for c in FCOLS))

print("\n" + "=" * 78 + "\n=== FACTOR SPANNING: FF5 + MOM + STR (Newey-West HAC) ===")
spanning(dm_g, "DM  (reversal sleeve)")
spanning(mom_stream, "MOM (trend regressor)")
print("[done]", flush=True)

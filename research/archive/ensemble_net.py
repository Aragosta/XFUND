#!/usr/bin/env python3
"""GO-TO ENSEMBLE, NET of the standard tiered cost model (tiered_transaction_costs + tiered_borrow_fees).
DM costed on its monthly $-volume; MOM costed on $-volume mapped from the broad daily download (the
750-daily checkpoint has no volume). Reports GROSS and NET side by side; equal-weight combine (the winner)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRegressor
import deep_momentum_xgb as d
from features import make_features, MOM_WINDOWS
import BACKTEST

N_SEEDS = 3
def perf(r):
    r = pd.Series(r).dropna(); ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return ann, (r.mean() * 12) / vol if vol > 0 else np.nan, mdd
def costed(gross, W, tc, bf):   # gross: {t:r}, W: {t:weight Series}; returns net {t:r}
    net, prev = {}, None
    for t in sorted(gross.index):
        w = W[t]
        if prev is not None:
            dw = w.subtract(prev, fill_value=0).abs()
            c = float((dw * tc.loc[t].reindex(dw.index).fillna(0.015)).sum())
            b = float((w.clip(upper=0).abs() * bf.loc[t].reindex(w.index).fillna(0.05)).sum()) / 12.0
        else: c = b = 0.0
        net[t] = gross[t] - c - b; prev = w
    return pd.Series(net)

# ═══ DM sleeve ═══
print("[DM] load + weights ...", flush=True)
pm, rm, sm = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
elig, short = d.compute_eligibility(pm, sm, min_dollar_vol_pct=0.30, min_dollar_vol_abs=1e6)
pnl = d._build_pnl_prices(pm); rpnl = pnl.pct_change()
tc_dm = BACKTEST.tiered_transaction_costs(sm); bf_dm = BACKTEST.tiered_borrow_fees(sm)
first_pred = d.MIN_TRAIN_YRS * 12 + max(MOM_WINDOWS) + 1
ffd = d._ffd_from_training_window(pm, first_pred)
for m in list(ffd): ffd[m] = ffd[m].reindex(rm.index)
dm_w = d._generate_all_dm_weights(rm, sm, pm, q=d.TOP_Q, min_train_months=d.MIN_TRAIN_YRS * 12,
        max_train_months=120, n_seeds=N_SEEDS, use_ffd=True, ffd_scores=ffd,
        portfolio="ls", eligible=elig, shortable=short)
dm = dm_w["ret"]
dm_g, dm_W = {}, {}
for t in dm.index:
    if t not in pnl.index: continue
    p = pnl.index.get_loc(t)
    if p + 1 >= len(pnl.index): break
    dm_g[t] = float((dm.loc[t] * rpnl.iloc[p + 1]).sum(skipna=True)); dm_W[t] = dm.loc[t]
dm_g = pd.Series(dm_g); dm_n = costed(dm_g, dm_W, tc_dm.reindex(dm.index), bf_dm.reindex(dm.index))

# ═══ MOM sleeve ═══
print("[MOM] features + $vol ...", flush=True)
px = pd.read_parquet("tiingo_daily_checkpoint.parquet").sort_index()
rets_d = px.pct_change(); vol_d = rets_d.ewm(span=63, min_periods=20).std()
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
mret = px.reindex(me).pct_change(); T = len(me)
eligm = (px.reindex(me) > 1.0) & (px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill") > 0.9)
# $-volume for MOM names from the broad daily download
cb = pd.read_parquet("tiingo_daily_close.parquet"); vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex_like(cb)
dvm = (cb * vb).resample("ME").sum(); dvm.index = dvm.index.to_period("M")
mom_dvol = dvm.reindex(pd.PeriodIndex(me, freq="M")); mom_dvol.index = me
mom_dvol = mom_dvol.reindex(columns=px.columns)
tc_mom = BACKTEST.tiered_transaction_costs(mom_dvol); bf_mom = BACKTEST.tiered_borrow_fees(mom_dvol)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")
def zc(s): return (s - s.mean()) / (s.std() + 1e-9)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp); PF = list(pf.keys())
pool = {}
for k in range(13, T - 3):
    dt = me[k]; P = pd.DataFrame({c: pf[c].loc[dt] for c in PF}); pn = mret.iloc[k + 1]
    ok = P.notna().all(axis=1) & mret.iloc[k+1].notna() & mret.iloc[k+2].notna() & mret.iloc[k+3].notna() & eligm.loc[dt].fillna(False)
    idx = P.index[ok.values]
    if len(idx) < 50: continue
    Y = np.column_stack([zc(mret.iloc[k+h].reindex(idx)).values for h in (1, 2, 3)])
    pool[k] = dict(P=P.loc[idx], Y=Y, pnl=pn.reindex(idx), dt=dt)
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)
print("[MOM] walk-forward ...", flush=True)
BASEP = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)
mom_g, mom_W = {}, {}; ms = None
for i in range(fp, len(keys)):
    k = keys[i]; dt = pool[k]["dt"]
    if dt.month == 1 or ms is None:
        tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt][-120:]
        if len(tr) >= 36:
            X = pd.concat([pool[t]["P"] for t in tr]).values; Y = np.vstack([pool[t]["Y"] for t in tr])
            ms = [XGBRegressor(**BASEP, multi_strategy="multi_output_tree", random_state=s).fit(X, Y) for s in range(N_SEEDS)]
    if ms is None: continue
    sc = np.mean([m.predict(pool[k]["P"].values).mean(1) for m in ms], axis=0)
    s = pd.Series(sc, index=pool[k]["pnl"].index); n = max(1, int(len(s) * 0.10))
    iv = (1 / (vol_d.loc[dt].reindex(s.index) * np.sqrt(21))).clip(upper=50)
    w = pd.Series(0.0, index=s.index); lo, sh = s.nlargest(n).index, s.nsmallest(n).index
    w[lo] = iv[lo] / iv[lo].sum(); w[sh] = -iv[sh] / iv[sh].sum()
    mom_g[dt] = float((w * pool[k]["pnl"]).sum()); mom_W[dt] = w
mom_g = pd.Series(mom_g); mom_n = costed(mom_g, mom_W, tc_mom, bf_mom)

# ═══ align + combine (equal-weight) gross & net ═══
def align(a, b):
    ap = a.copy(); ap.index = a.index.to_period("M"); bp = b.copy(); bp.index = b.index.to_period("M")
    return pd.DataFrame({"DM": ap, "MOM": bp}).dropna()
G = align(dm_g, mom_g); Ng = align(dm_n, mom_n)
print("\n" + "=" * 60)
print(f"=== ENSEMBLE: GROSS vs NET (tiered costs), {len(Ng)} months ===")
print(f"{'strategy':26}{'ann':>8}{'SR':>7}{'maxDD':>9}   {'annNET':>8}{'SRnet':>7}{'DDnet':>9}")
for nm, g, nn in [("DM (reversal)", G["DM"], Ng["DM"]), ("MOM (trend)", G["MOM"], Ng["MOM"]),
                  ("Combine equal-weight", G.mean(1), Ng.mean(1))]:
    ag, sg, mg = perf(g); an, sn, mn = perf(nn)
    print(f"{nm:26}{ag:>8.1%}{sg:>7.2f}{mg:>9.1%}   {an:>8.1%}{sn:>7.2f}{mn:>9.1%}")
print(f"\ncorr(DM,MOM) net = {Ng.corr().iloc[0,1]:.2f}")
print("[done]", flush=True)

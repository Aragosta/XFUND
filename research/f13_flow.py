#!/usr/bin/env python3
"""f13_flow.py — adjudicate the framework question with data: does 13F FLOW/CONVICTION (changes, not levels)
predict RETURNS or only the 2nd MOMENT, and where (large vs small cap)?

Build a quarterly per-ticker 13F panel from raw holdings: breadth = #institutions holding, value = total $ held.
Signals: level (breadth, log-value) and CHANGES (Δbreadth = net initiations, Δlog-value = flow). PIT: a period-end
quarter is public ~45d later -> earns from the following month. Test each signal's cross-sectional IC on (a) fwd
1-month return, (b) fwd 3-month return, (c) fwd realized vol (2nd moment). Split large/small cap. This decides:
is our 'no alpha' a LEVELS-only artifact (framework wrong) or is 13F genuinely a 2nd-moment/risk signal?"""
import warnings; warnings.filterwarnings("ignore")
import glob, json, numpy as np, pandas as pd

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2012-01-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0)
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6)
dvol = px.pct_change().rolling(21).std().resample("ME").last().reindex(me)*np.sqrt(21)                  # ~monthly realized vol
fwd1 = mret.shift(-1); fwd3 = (m_px.shift(-3)/m_px - 1.0).where(lambda z: z<2.0); fvol1 = dvol.shift(-1)  # forward targets

# ---- build quarterly 13F panel: (period, ticker) -> n_holders, value ----
c2t = json.load(open("data/13f/cusip_ticker.json")); up = {c.upper(): c for c in px.columns}
rows = []
for f in sorted(glob.glob("data/13f/parsed/*.parquet")):
    d = pd.read_parquet(f, columns=["cik","period","cusip","value"])
    d["tic"] = d["cusip"].str.slice(0,9).map(c2t); d = d.dropna(subset=["tic"])
    d["tic"] = d["tic"].str.upper().map(up); d = d.dropna(subset=["tic"])
    g = d.groupby(["period","tic"]).agg(hold=("cik","nunique"), val=("value","sum")).reset_index()
    rows.append(g)
P = pd.concat(rows, ignore_index=True)
P["pend"] = pd.to_datetime(P["period"], format="%d-%b-%Y", errors="coerce"); P = P.dropna(subset=["pend"])
P = P[P["pend"] >= "2011-06-01"]
periods = sorted(P["pend"].unique())
HOLD = P.pivot_table(index="pend", columns="tic", values="hold").reindex(columns=px.columns)
VAL  = P.pivot_table(index="pend", columns="tic", values="val").reindex(columns=px.columns)
print(f"[13f] {len(periods)} quarters {pd.Timestamp(periods[0]).date()}..{pd.Timestamp(periods[-1]).date()}, {HOLD.notna().any().sum()} tickers ever held", flush=True)

# quarterly signals -> forward-fill to monthly, PIT lag one quarter (public ~45d after period end)
def to_monthly_pit(Q):
    Qs = Q.copy(); Qs.index = [pd.Timestamp(p) + pd.Timedelta(days=50) for p in Q.index]                # availability date
    return Qs.reindex(me, method="ffill")
lvl_breadth = to_monthly_pit(np.log1p(HOLD))
lvl_value   = to_monthly_pit(np.log1p(VAL))
d_breadth   = to_monthly_pit(HOLD.diff() / (HOLD.shift(1)+1))                                            # net initiation rate
d_value     = to_monthly_pit(np.log(VAL) - np.log(VAL.shift(1)))                                         # $ flow (log change)

def xs_ic(sig, fwd, mask):
    ics = []
    for d in me:
        s = sig.loc[d].where(mask.loc[d]); y = fwd.loc[d]
        df = pd.DataFrame({"s":s,"y":y}).dropna()
        if len(df) < 40: continue
        ics.append(df["s"].corr(df["y"], method="spearman"))
    ics = np.array(ics); return np.nanmean(ics), np.nanmean(ics)/(np.nanstd(ics)/np.sqrt(len(ics))+1e-9), len(ics)

# cap split: large = top half by mdv among eligible each month
med = mdv.where(elig).median(axis=1); large = elig & mdv.ge(med, axis=0); small = elig & mdv.lt(med, axis=0)
sigs = {"breadth level":lvl_breadth, "value level":lvl_value, "Δbreadth (initiations)":d_breadth, "Δvalue (flow)":d_value}
print("\n13F SIGNAL -> IC (Spearman, leak-free). fwd1=next-mo ret, fwd3=3-mo ret, fvol=next-mo vol")
print(f"  {'signal':24}{'universe':10}{'IC_ret1':>9}{'t':>6}{'IC_ret3':>9}{'t':>6}{'IC_vol':>9}{'t':>6}")
for nm, sig in sigs.items():
    for un, mask in [("all",elig),("large",large),("small",small)]:
        i1,t1,_ = xs_ic(sig, fwd1, mask); i3,t3,_ = xs_ic(sig, fwd3, mask); iv,tv,n = xs_ic(sig, fvol1, mask)
        print(f"  {nm:24}{un:10}{i1:>+9.3f}{t1:>+6.1f}{i3:>+9.3f}{t3:>+6.1f}{iv:>+9.3f}{tv:>+6.1f}")

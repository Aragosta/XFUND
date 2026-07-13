#!/usr/bin/env python3
"""RISKENGINE.py — the foundational SECOND-MOMENT / construction layer. One engine every sleeve plugs into.

WHY THIS EXISTS (the whole research program distilled): the 1st moment (direction) is a wall — untimeable,
un-adaptable, everything that tries to time it converges to static. The 2nd moment (vol / covariance / dispersion)
is forecastable (vol R^2 ~55%) and is where ALL our data has power — price, 13F (vol IC -0.3), news (dispersion
R^2 +0.05, attention t-5). So the edge is CONSTRUCTION, not more signals. This engine unifies the construction
machinery that was scattered across BETANEUT / ERC / vol-target / dispersion so current AND future sleeves share it.

THREE HARD RULES the research earned (enforced by the API, not optional):
  1. Risk model for SIZING, never for neutralization beyond market beta  (multi-factor neut ate the alpha:
     momentum IS a factor -> projecting it out removes the return with the risk).
  2. Never for TIMING / regime-switching  (regime + adaptive allocation both converge to static; `regime()`
     returns a DIAGNOSTIC, it does not gate alpha).
  3. New data (13F, news) enters as RISK inputs, not alpha sleeves  (they carry 2nd-moment signal, no direction).

API (all leak-free — every output at date t uses only data < t):
  eng = RiskEngine(use_13f=False, use_news=False)
  eng.idio_vol()            -> DataFrame[dates x names]  per-name vol forecast (price [+13F +news])
  eng.turbulence()          -> Series[dates] in [0,1]    market 2nd-moment regime (DIAGNOSTIC only)
  eng.factor_cov(d, names)  -> (Sigma, names)            statistical factor covariance for SIZING
  eng.neutralize(W)         -> DataFrame                 market-beta strip (rule 1) -> gross `target_gross`
  eng.size(W, mode)         -> DataFrame                 risk-size names by idio_vol / factor_cov
  eng.construct(W_raw, ...) -> DataFrame                 sleeve -> book pipeline (neutralize [+ size])
  eng.allocate(sleeve_rets) -> DataFrame                 static ERC across sleeves (rule 2: static)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import BETANEUT, ERC


class RiskEngine:
    def __init__(self, target_gross=2.0, vol_halflife=6, cov_win=36, beta_win=60,
                 use_13f=False, use_news=False, price="tiingo_daily_close.parquet",
                 volume="tiingo_daily_volume.parquet", start="2010-12-01"):
        self.target_gross = target_gross; self.cov_win = cov_win
        px = pd.read_parquet(price).sort_index()
        drop = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"])
        self.px = px = px.loc[:, ~drop]
        vb = pd.read_parquet(volume).reindex(columns=px.columns).reindex(index=px.index)
        self.me = me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
        self.me = me = me[me >= pd.Timestamp(start)]
        self.m_px = m_px = px.reindex(me)
        self.mret = mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0)
        self.synth = (1 + mret.fillna(0.0)).cumprod()
        self.mdv = mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill")
        cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
        self.elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6)
        self.BETA = BETANEUT.rolling_beta(mret, self.elig, bw=beta_win)
        # base per-name vol: EWMA of |monthly return|-driven realized vol, leak-free (shift 1)
        rv = px.pct_change().rolling(21).std().resample("ME").last().reindex(me)*np.sqrt(21)
        self._base_vol = rv.ewm(halflife=vol_halflife, min_periods=3).mean().shift(1)
        self.use_13f, self.use_news = use_13f, use_news
        self._breadth = self._load_13f() if use_13f else None
        self._attn = self._load_news() if use_news else None

    # ---------- optional 2nd-moment data enrichments (rule 3) ----------
    def _load_13f(self):
        import glob, json
        try:
            c2t = json.load(open("data/13f/cusip_ticker.json")); up = {c.upper(): c for c in self.px.columns}
            rows = []
            for f in sorted(glob.glob("data/13f/parsed/*.parquet")):
                d = pd.read_parquet(f, columns=["cik","period","cusip"]); d["tic"]=d["cusip"].str.slice(0,9).map(c2t)
                d = d.dropna(subset=["tic"]); d["tic"]=d["tic"].str.upper().map(up); d=d.dropna(subset=["tic"])
                rows.append(d.groupby(["period","tic"]).agg(hold=("cik","nunique")).reset_index())
            P = pd.concat(rows); P["pend"]=pd.to_datetime(P["period"], format="%d-%b-%Y", errors="coerce"); P=P.dropna(subset=["pend"])
            H = P.pivot_table(index="pend", columns="tic", values="hold").reindex(columns=self.px.columns)
            Hs = H.copy(); Hs.index = [pd.Timestamp(p)+pd.Timedelta(days=50) for p in H.index]     # PIT filing lag
            return np.log1p(Hs.reindex(self.me, method="ffill"))                                   # breadth: higher -> lower vol
        except Exception as e:
            warnings.warn(f"13F load failed ({e}); price-only vol"); return None

    def _load_news(self):
        try:
            g = pd.read_parquet("data/fnspid/sent_monthly.parquet")
            g["d"] = g["per"].map({d.to_period("M").__str__(): d for d in self.me}); g = g.dropna(subset=["d"])
            cnt = g.pivot_table(index="d", columns="sym", values="count").reindex(index=self.me, columns=self.px.columns)
            return np.log1p(cnt).shift(1)                                                          # attention: higher -> higher vol
        except Exception as e:
            warnings.warn(f"news load failed ({e}); price-only vol"); return None

    # ---------- 2nd-moment forecasts ----------
    def idio_vol(self):
        """Per-name vol forecast (leak-free). Base = EWMA realized vol; optionally cross-sectionally blend
        13F breadth (-) and news attention (+) as small tilts to the base rank."""
        V = self._base_vol.copy()
        if self._breadth is None and self._attn is None:
            return V
        def zc(df): d = df.replace([np.inf,-np.inf], np.nan); return d.sub(d.mean(1),axis=0).div(d.std(1)+1e-9,axis=0)
        adj = pd.DataFrame(0.0, index=V.index, columns=V.columns)
        if self._breadth is not None: adj = adj.add(-0.10*zc(self._breadth), fill_value=0.0)       # more owners -> lower vol
        if self._attn    is not None: adj = adj.add(+0.10*zc(self._attn),    fill_value=0.0)       # more news  -> higher vol
        return (V * (1.0 + adj.clip(-0.5, 0.5))).where(V.notna())

    def turbulence(self):
        """Market 2nd-moment regime in [0,1] (expanding percentile). DIAGNOSTIC ONLY — does NOT gate alpha (rule 2)."""
        disp = self.mret.where(self.elig).std(axis=1)
        mvol = self._base_vol.mean(axis=1)
        z = lambda s: (s - s.expanding(12).mean())/(s.expanding(12).std()+1e-9)
        turb = (z(disp) + z(mvol))/2
        return turb.expanding(12).apply(lambda a: (a.iloc[-1] > a).mean())

    def factor_cov(self, d, names, K=10, shrink=0.15):
        """Statistical K-factor covariance for a set of names at date d (leak-free), for SIZING only (rule 1)."""
        i = list(self.me).index(d)
        if i < self.cov_win: return None, names
        H = self.mret.iloc[i-self.cov_win+1:i+1][names].dropna(axis=1, thresh=self.cov_win-2)
        cols = list(H.columns); R = H.fillna(0.0).values; R = R - R.mean(0, keepdims=True)
        U,S,Vt = np.linalg.svd(R, full_matrices=False); K = min(K, len(S)-1)
        F = U[:, :K]*S[:K]; B = Vt[:K].T                                                           # factor rets, loadings
        sys = B @ np.cov(F.T) @ B.T                                                                # systematic cov
        spec = np.var(R - F @ B.T, axis=0)                                                         # specific var (diag)
        Sigma = sys + np.diag(spec + 1e-6)
        Sigma = (1-shrink)*Sigma + shrink*np.diag(np.diag(Sigma))                                  # shrink for stability
        return Sigma, cols

    # ---------- construction ops (rule 1) ----------
    def neutralize(self, W):
        """Market-beta strip -> gross target_gross. The ONE neutralization that works (more factors eat alpha)."""
        Wn = BETANEUT.betaneut(W.reindex(columns=self.px.columns).fillna(0.0), self.BETA)
        return Wn * (self.target_gross/2.0)                                                        # BETANEUT renorms to 2.0

    def size(self, W, mode="invvol"):
        """Risk-size names within a book by idio_vol (invvol) or factor_cov. Preserves each date's dollar-neutral
        L/S structure; renorms to target_gross.
        VERDICT (tested on MOM): risk-sizing HURTS rank-based sleeves (net SR 0.53 -> 0.12) — MOM's alpha is in the
        equal-weight decile RANKING, and inverse-vol re-weighting concentrates in low-vol names regardless of signal,
        eating the alpha (same failure mode as multi-factor neutralization). DEFAULT OFF. Exposed for future sleeves
        whose alpha is magnitude- not rank-based; for our momentum book, neutralize-only is the champion."""
        V = self.idio_vol(); out = pd.DataFrame(0.0, index=W.index, columns=W.columns)
        for d in W.index:
            w = W.loc[d];
            if w.abs().sum() < 1e-9: continue
            if mode == "invvol":
                iv = (1.0/V.loc[d]).replace([np.inf,-np.inf], np.nan)
                s = (np.sign(w) * iv).where(w.abs()>0).fillna(0.0)                                 # keep side, scale by 1/vol
            else:
                s = w
            L = s.clip(lower=0); S = s.clip(upper=0); gl, gs = L.sum(), -S.sum()
            if gl>1e-9 and gs>1e-9:
                s = L*(1.0/gl) + S*(1.0/gs)                                                        # re-balance L/S to dollar-neutral (S<0)
            g = s.abs().sum(); out.loc[d] = s*(self.target_gross/g) if g>0 else s
        return out

    def construct(self, W_raw, neutralize=True, size=False, size_mode="invvol"):
        """Full sleeve->book pipeline. Default = the proven champion (neutralize only). `size` is the open lever."""
        W = W_raw.reindex(columns=self.px.columns).fillna(0.0)
        if size: W = self.size(W, mode=size_mode)
        if neutralize: W = self.neutralize(W)
        return W

    # ---------- sleeve allocation (rule 2: STATIC) ----------
    def allocate(self, sleeve_rets, method="erc", win=36):
        """Static risk-parity across sleeves (proven to beat adaptive). Thin wrapper on ERC.expanding_alloc."""
        W = ERC.expanding_alloc(sleeve_rets, method=method, win=win)
        return W, ERC.combine(sleeve_rets, W)


if __name__ == "__main__":
    import pickle, BACKTEST
    eng = RiskEngine()
    print("="*90); print("RISKENGINE self-test — validate each component on the MOM book")
    tc = BACKTEST.tiered_transaction_costs(eng.mdv); bf = BACKTEST.tiered_borrow_fees(eng.mdv)
    def net(W):
        r = BACKTEST.backtest(W.fillna(0.0), eng.synth, freq=12, lag=0,
                              signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9], transaction_cost=tc, borrow_fee=bf)
        s = pd.Series(r["returns"]); s.index = pd.DatetimeIndex(s.index)
        s = s[(s.index>="2011-06-01")&(s.index<"2023-01-01")].dropna(); e=(1+s).cumprod()
        return s.mean()/s.std()*np.sqrt(12), (e/e.cummax()-1).min(), r["ann_turnover"]

    MOM = pickle.load(open("/tmp/mom_weights.pkl","rb")); MOM.index = pd.DatetimeIndex(MOM.index)
    MOM = MOM.reindex(index=eng.me, columns=eng.px.columns).fillna(0.0)
    print("\n  [neutralize] the proven path (market-beta strip):")
    for lbl, W in [("raw (dollar-neutral)", eng.construct(MOM, neutralize=False)),
                   ("neutralize only", eng.construct(MOM)),
                   ("neutralize + invvol size (OPEN LEVER)", eng.construct(MOM, size=True))]:
        sr, dd, tn = net(W); print(f"    {lbl:38} net SR {sr:>5.2f}  maxDD {dd:>6.1%}  turn {tn:>4.1f}")

    print("\n  [idio_vol] forecast quality — IC vs next-month realized vol:")
    V = eng.idio_vol(); rvf = eng._base_vol.shift(-1)                                              # realized next month
    ics = [V.loc[d].corr(rvf.loc[d], method="spearman") for d in eng.me if eng.elig.loc[d].sum()>40]
    print(f"    mean IC(vol_forecast, next-vol) = {np.nanmean(ics):+.3f}")
    print("\n  [turbulence] regime diagnostic (last 6 months):")
    print("    " + "  ".join(f"{d.strftime('%Y-%m')}:{v:.2f}" for d,v in eng.turbulence().dropna().tail(6).items()))
    print("\n[done] RiskEngine validated.")

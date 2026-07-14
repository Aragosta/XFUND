#!/usr/bin/env python3
"""DATAHUB.py — the coherent DATA layer, DAILY-NATIVE. Single source of truth every sleeve draws from.

Grid: DAILY is the fundamental unit (reversion/news/daily construction live here); MONTHLY is just a resample
(MOM/DM). Fundamentals update quarterly and are PIT-forward-filled onto whichever grid you ask for.

THE BUG THIS KILLS: Tiingo `close` is split-ADJUSTED, EDGAR shares are AS-REPORTED (pre-split) -> mcap wrong for
any name that split after t (growth splitters look like value). Fix: reconstruct cumulative split factor from
JUMPS in the reported-shares series (done once, at report level), put shares on the CURRENT (adjusted-price)
basis -> mcap = adj_price * adj_shares is split-COHERENT by construction (verified: NVDA B/M 0.225 -> 0.023).
[Full fidelity wants a Tiingo re-pull of raw close + splitFactor; this is the API-key-free interim.]

API:
  hub = DataHub()
  hub.px_d, hub.vol_d, hub.ret_d, hub.adv_d, hub.sma_d(200)      # daily panels
  hub.me, hub.m_px, hub.mret, hub.synth, hub.mdv                 # monthly (resampled) for MOM/DM + BACKTEST
  hub.mcap('daily'|'monthly'), hub.bm(g), hub.ep(g), hub.gpa(g), hub.roe(g), hub.shares(g)   # split-coherent
  hub.elig('liquid'|'relaxed', grid)                             # one universe definition, shared
  hub.M(daily_panel)                                            # resample any daily panel -> month-end
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_CLEAN = np.array([1.5,2,3,4,5,6,7,8,10,12,15,20,25,30]); _CLEAN = np.concatenate([_CLEAN, 1/_CLEAN])
def _snap_split(r):
    if not np.isfinite(r) or 0.72 < r < 1.4: return 1.0
    j = np.argmin(np.abs(_CLEAN - r)); return float(_CLEAN[j]) if abs(_CLEAN[j]-r)/_CLEAN[j] < 0.12 else 1.0

class DataHub:
    def __init__(self, start="2010-12-01", edgar="data/edgar/facts.parquet", min_days=250):
        px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
        drop = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"])
        px = px.loc[:, ~drop]; px = px.loc[:, px.notna().sum() >= min_days]           # trim never-traded junk
        self.px_d = px; self.days = px.index
        self.vol_d = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
        # HONEST RETURNS: prices are split-ADJUSTED, so a huge move is a REAL move (a squeeze), not an artifact.
        # The old guards (>=+100%/mo, >=+50%/day -> NaN -> 0) silently ERASED short squeezes — and our alpha is
        # short-side-heavy, so they flattered the book enormously (VQ SR 2.06 -> 0.76 once squeezes are paid for).
        # Only absurd moves are treated as data errors now.
        self.ret_d = px.pct_change(fill_method=None).where(lambda z: z.abs() < 2.0)      # >200%/day = data error
        self.adv_d = (px*self.vol_d).rolling(21, min_periods=10).mean()               # 21d avg dollar volume
        self._sma = {}
        # monthly resample (for MOM/DM + the monthly BACKTEST engine)
        self.me = me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
        self.me = me = me[me >= pd.Timestamp(start)]
        self.m_px = px.reindex(me); self.mret = self.m_px.pct_change(fill_method=None).where(lambda z: z < 10.0)  # >1000%/mo = data error
        self.synth = (1 + self.mret.fillna(0.0)).cumprod()
        _dvm = (px*self.vol_d).resample("ME").sum(); _dvm.index = _dvm.index.to_period("M")   # PERIOD align (not ffill:
        self.mdv = _dvm.reindex(pd.PeriodIndex(me, freq="M")); self.mdv.index = me             # calendar-end label vs trading-day me)
        self.cov_m = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
        self._load_edgar(edgar); self._cache = {}
        # RAW (unadjusted) monthly close -> correct mcap = raw_price × as-reported shares (both same split basis)
        self.raw_m_px = None
        try:
            rp = pd.read_parquet("data/tiingo_raw_monthly.parquet"); rp.index = pd.to_datetime(rp.index)
            rp.index = rp.index.to_period("M"); ra = rp.reindex(pd.PeriodIndex(self.me, freq="M")); ra.index = self.me
            self.raw_m_px = ra.reindex(columns=self.px_d.columns)
        except Exception: pass
        self.spy_m = self._load_spy()                                            # SPY buy-and-hold benchmark (monthly return)
        try:                                                                     # SIC industry (2-digit sector) per ticker
            _sic = pd.read_parquet("data/edgar/sic.parquet")
            self.sector = pd.Series(_sic.set_index("ticker")["sector2"]).reindex(self.px_d.columns)
        except Exception: self.sector = None
    def fund(self, concept, grid="monthly"): return self._get(concept, grid)     # any PIT fundamental concept

    def _load_spy(self):
        """SPY buy-and-hold monthly return, aligned to `me` — the market benchmark (we want to be up when SPY is up)."""
        import os
        for p in ("data/spy.parquet", "/tmp/spy.parquet"):
            try:
                if os.path.exists(p):
                    s = pd.read_parquet(p); s = s[s.columns[0]] if hasattr(s, "columns") else s
                    m = s.resample("ME").last().pct_change(); m.index = pd.DatetimeIndex(m.index).to_period("M")
                    out = m.reindex(pd.PeriodIndex(self.me, freq="M")); out.index = self.me; return out
            except Exception: pass
        return None

    def sma_d(self, w=200):
        if w not in self._sma: self._sma[w] = self.px_d.rolling(w, min_periods=int(w*0.75)).mean()
        return self._sma[w]
    def M(self, daily_panel):                                                        # daily -> month-end
        return daily_panel.reindex(self.days).resample("ME").last().reindex(self.me)

    # ---------- fundamentals: split-coherence done ONCE at report level ----------
    def _load_edgar(self, edgar):
        try:
            f = pd.read_parquet(edgar).dropna(subset=["val"]).copy(); f["end"] = pd.to_datetime(f["end"])
        except Exception as e:
            warnings.warn(f"EDGAR load failed ({e}); fundamentals unavailable"); self._f = None; return
        LAG = {"book":90,"shares":90,"ni":120,"assets":90,"rev":120,"cogs":120}
        f["avail"] = f["end"] + pd.to_timedelta(f["concept"].map(LAG).fillna(90), unit="D")
        f = f[f["ticker"].isin(self.px_d.columns)]
        # keep AS-REPORTED shares (concept 'shares_rep') for RAW-price mcap; build split-COHERENT ('shares') for
        # the fallback (adjusted-price) mcap when raw price is unavailable.
        sh = f[f.concept == "shares"].sort_values(["ticker","avail","end"]).copy()
        sh_rep = sh.copy(); sh_rep["concept"] = "shares_rep"                          # as-reported (pairs with raw price)
        newval = sh["val"].values.astype(float)
        for tk, idx in sh.groupby("ticker").indices.items():
            v = sh["val"].values[idx];
            if len(v) < 2: continue
            fac = np.ones(len(v))
            for i in range(1, len(v)): fac[i] = _snap_split(v[i]/v[i-1] if v[i-1] else 1.0)
            future = np.append(np.cumprod(fac[::-1])[::-1][1:], 1.0)                  # splits AFTER each report
            newval[idx] = v * future
        sh["val"] = newval; f = pd.concat([f[f.concept != "shares"], sh, sh_rep], ignore_index=True)
        self._f = f

    def _pit(self, concept, grid):
        """latest value available (period-end + reporting lag) at each grid date; forward-filled."""
        idxgrid = self.days if grid == "daily" else self.me
        out = pd.DataFrame(np.nan, index=idxgrid, columns=self.px_d.columns)
        if self._f is None: return out
        sub = self._f[self._f.concept == concept]
        for tk, g in sub.groupby("ticker"):
            g = g.sort_values(["avail","end"]).drop_duplicates("avail", keep="last")
            k = np.searchsorted(g["avail"].values, idxgrid.values, side="right") - 1
            out[tk] = np.where(k >= 0, g["val"].values[k.clip(min=0)], np.nan)
        return out

    def _get(self, key, grid):                                                       # lazy + cached
        ck = (key, grid)
        if ck in self._cache: return self._cache[ck]
        px = self.px_d if grid == "daily" else self.m_px
        RAW = ("book","shares","ni","assets","rev","cogs","ocf","debt_lt","inventory","receivables",
               "assets_cur","liab_cur","cash","capex","dividends","gross_profit")
        if key in RAW: v = self._pit(key, grid)
        elif key == "mcap":
            # TESTED: raw_price × as-reported shares is WORSE (IC(B/M) +0.022 -> -0.007) — a split hits price
            # immediately but the new share count lags to the next 10-Q, so mcap blows up ~1 quarter after every
            # split. adjusted price × split-COHERENT shares keeps both on one basis -> cleaner. Raw kept for the
            # future splitFactor-based build (daily splitFactor from the OHLC pull -> correct daily raw mcap).
            v = self.shares(grid) * px                                               # adjusted × split-coherent (better)
        elif key == "bm":   v = (self.book(grid) / self.mcap(grid)).where(self.book(grid) > 0)
        elif key == "ep":   v = self._get("ni", grid) / self.mcap(grid)
        elif key == "gpa":  v = ((self._get("rev",grid)-self._get("cogs",grid))/self._get("assets",grid)).where(self._get("assets",grid)>0)
        elif key == "roe":  v = (self._get("ni",grid)/self.book(grid)).where(self.book(grid) > 0)
        self._cache[ck] = v; return v
    def shares(self, g="monthly"): return self._get("shares", g)
    def book(self, g="monthly"):   return self._get("book", g)
    def mcap(self, g="monthly"):   return self._get("mcap", g)
    def bm(self, g="monthly"):     return self._get("bm", g)
    def ep(self, g="monthly"):     return self._get("ep", g)
    def gpa(self, g="monthly"):    return self._get("gpa", g)
    def roe(self, g="monthly"):    return self._get("roe", g)

    # ---------- daily factor construction + backtest (the baseline engine) ----------
    def decile_book(self, sig, uni="liquid", q=0.10, reversal=False, grid="daily"):
        """Cross-sectional decile L/S book, dollar-neutral gross 2. reversal -> long LOW signal."""
        elig = self.elig(uni, grid); S = sig.where(elig); R = S.rank(axis=1, pct=True)
        long_ = (R <= q) if reversal else (R >= 1-q); short = (R >= 1-q) if reversal else (R <= q)
        L = long_.astype(float); Sh = short.astype(float)
        L = L.div(L.sum(1).replace(0,np.nan), axis=0); Sh = Sh.div(Sh.sum(1).replace(0,np.nan), axis=0)
        return (L - Sh).fillna(0.0)

    def backtest_daily(self, W_target, H=5, lag=2, cost_bps=5.0):
        """Honest daily engine: overlapping H-day tranches, strict next-close exec (signal[d]->earn d+2 => lag=2),
        per-side bps on daily turnover. Returns (gross, net, avg_daily_turnover)."""
        Wbook = W_target.rolling(H, min_periods=1).mean()
        gross = (Wbook.shift(lag) * self.ret_d).sum(axis=1)
        turn = (Wbook - Wbook.shift(1)).abs().sum(axis=1)
        net = gross - (cost_bps/1e4) * turn.shift(lag)
        return gross.dropna(), net.dropna(), turn.mean()

    @staticmethod
    def stats(x, ann=252):
        x = x.dropna(); e = (1+x).cumprod()
        return dict(SR=x.mean()/x.std()*np.sqrt(ann), ann=(1+x).prod()**(ann/len(x))-1, maxDD=(e/e.cummax()-1).min())

    # ---------- delisting-family data engineering (GLOBAL — every sleeve sees delistings cost money) ----------
    def _inject_delist(self, px, r=-0.30):
        px = px.copy(); idx = px.index; last = len(idx) - 1
        for col in px.columns:
            s = px[col]; lv = s.last_valid_index()
            if lv is None: continue
            pos = idx.get_loc(lv)
            if pos < last: px.iloc[pos + 1, px.columns.get_loc(col)] = s.iloc[pos] * (1.0 + r)
        return px
    def delisted_prices(self, grid="monthly"):
        """Monthly (or daily) adj-close with a synthetic -30% the period after a name stops trading."""
        k = ("dlpx", grid)
        if k not in self._cache: self._cache[k] = self._inject_delist(self.m_px if grid=="monthly" else self.px_d)
        return self._cache[k]
    def clean_returns(self, grid="monthly"):
        """Delisting-aware returns, cross-sectionally 1/99 winsorized (Han §4.1) — the modelling-return basis."""
        k = ("clret", grid)
        if k not in self._cache:
            raw = self.delisted_prices(grid).pct_change()
            self._cache[k] = raw.clip(raw.quantile(0.01, axis=1), raw.quantile(0.99, axis=1), axis=0)
        return self._cache[k]
    def dollar_size(self, grid="monthly"):
        """Size proxy = adj-close × period-end volume (distinct from mdv's daily-sum), forward-filled."""
        k = ("dsz", grid)
        if k not in self._cache:
            if grid == "monthly":
                v = self.vol_d.resample("ME").last(); v.index = v.index.to_period("M")            # PERIOD align to me
                vol = v.reindex(pd.PeriodIndex(self.me, freq="M")); vol.index = self.me
                self._cache[k] = (self.m_px * vol).ffill()
            else:
                self._cache[k] = (self.px_d * self.vol_d).ffill()
        return self._cache[k]
    def pnl(self, grid="monthly"):
        """HONEST synthetic PnL price grid: delisting + a DATA-ERROR guard only, then cumprod.
        The old version cross-sectionally 1/99-winsorized the PnL, which CAPPED short-squeeze losses — the book
        is short-side-heavy, so that silently erased the one risk that kills short books. Winsorizing is fine for
        MODELLING labels (see clean_returns) but must NEVER touch the PnL: a +300% squeeze must be paid in full."""
        k = ("pnl", grid)
        if k not in self._cache:
            px = self.delisted_prices(grid); r = px.pct_change().clip(lower=-0.95, upper=10.0)   # >1000% = data error
            self._cache[k] = (1.0 + r.fillna(0.0)).cumprod().where(px.notna())
        return self._cache[k]

    # ---------- one universe definition, shared by all sleeves ----------
    def elig(self, kind="liquid", grid="monthly"):
        if grid == "daily":
            base = (self.px_d > 5) & self.px_d.notna() & self.adv_d.notna()
            return base & (self.adv_d > 5e6) if kind == "liquid" else base & (self.adv_d > 5e5)
        base = (self.m_px > 5) & (self.cov_m > 0.9)
        if kind == "liquid":  return base & (self.mdv > 5e6)
        if kind == "relaxed": return (self.m_px > 3) & (self.cov_m > 0.8) & (self.mdv > 5e5)
        return base

if __name__ == "__main__":
    hub = DataHub()
    print("="*82); print("DATAHUB (daily-native) — self-test")
    print(f"  daily {hub.px_d.shape}  ({hub.days[0].date()}..{hub.days[-1].date()})   monthly {hub.m_px.shape}")
    d = hub.me[hub.me <= "2023-12-31"][-1]
    print(f"\n  split-coherent mcap @ {d.date()} (monthly grid):")
    mc, bm, gpa = hub.mcap("monthly"), hub.bm("monthly"), hub.gpa("monthly")
    for tk in ["NVDA","AAPL","TSLA","JPM","XOM","KO","BAC"]:
        if tk in mc.columns and np.isfinite(mc.loc[d,tk]):
            print(f"    {tk:5} mcap ${mc.loc[d,tk]/1e9:>8.0f}B   B/M {bm.loc[d,tk]:>6.3f}   GP/A {gpa.loc[d,tk]:>6.3f}")
    for k in ("liquid","relaxed"):
        em = hub.elig(k,"monthly"); print(f"  elig {k:8}: {int(em.sum(1).mean())} names/mo   with fundamentals: {int((bm.notna()&em).sum(1).mean())}/mo")
    print(f"  daily mcap check: shape {hub.mcap('daily').shape}, NVDA @ {hub.days[-1].date()} = ${hub.mcap('daily').iloc[-1].get('NVDA',np.nan)/1e9:.0f}B")
    print("[done] sleeves should `from DATAHUB import DataHub` instead of re-loading + re-deriving.")

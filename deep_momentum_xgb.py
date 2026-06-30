#!/usr/bin/env python3
"""
Deep Momentum XGBoost — Han (2022) proof of concept.

Two-step approach:
  1. XGBoost multiclass classifier predicts next-month return decile probabilities.
  2. Three reclassification scores (DPR / RET / SRP) re-rank stocks to filter
     out the bimodality-driven crash tail of momentum strategies.

Usage:
    python deep_momentum_xgb.py                        # synthetic data
    python deep_momentum_xgb.py --compare-tiingo       # Tiingo broad universe
    python deep_momentum_xgb.py sp500_stocks.csv srp   # CSV archive
"""
import sys
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from features import make_features, MOM_WINDOWS

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
N_SEEDS       = 20                  # ensemble size (paper uses 50-100; 20 balances speed vs. stability)
MIN_TRAIN_YRS = 10                  # minimum training history before first pred
TOP_Q         = 0.10                # long / short tail size

# Paper: "default hyperparameters, except for early stopping"
XGB_PARAMS = dict(
    num_class=10,
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=300,
    max_depth=3,
    learning_rate=0.10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    verbosity=0,
)

# ── 1. Data loading ───────────────────────────────────────────────────────────
def load_data(csv_path=None):
    """
    Returns (rets, size) DataFrames, each T × N (monthly).

    rets : monthly returns clipped to ±50%
    size : market-cap proxy (Close × Volume), used for size-decile features
    """
    if csv_path:
        try:
            df = (
                pd.read_csv(csv_path, parse_dates=["date"])
                .sort_values("date")
            )
            close  = df.pivot_table(index="date", columns="Name", values="close",
                                    aggfunc="last").resample("ME").last()
            volume = df.pivot_table(index="date", columns="Name", values="volume",
                                    aggfunc="last").resample("ME").last()
            rets   = close.pct_change().clip(-0.5, 0.5)
            size   = (close * volume).fillna(method="ffill")
            return rets, size
        except Exception as e:
            print(f"[warn] Cannot load {csv_path}: {e}. Using synthetic data.")

    # Synthetic fallback: 200 months × 300 stocks with a mild momentum factor
    rng    = np.random.default_rng(42)
    T, N   = 200, 300
    dates  = pd.date_range("2005-01", periods=T, freq="ME")
    stocks = [f"S{i:03d}" for i in range(N)]
    market = rng.standard_normal(T) * 0.04
    betas  = rng.uniform(0.5, 1.5, N)
    raw    = np.outer(market, betas) + rng.standard_normal((T, N)) * 0.06 + 0.003
    rets   = pd.DataFrame(raw.clip(-0.5, 0.5), index=dates, columns=stocks)
    size   = pd.DataFrame(rng.lognormal(10, 1.5, (T, N)), index=dates, columns=stocks)
    return rets, size


# ── 2. Feature engineering — imported from features.py ───────────────────────
# make_features(rets, size, t, ffd_scores=None) → DataFrame(N_stocks × features)

# ── 3. Label construction ─────────────────────────────────────────────────────
def _decile_labels(fwd: pd.Series) -> pd.Series:
    """
    Rank-based cross-sectional return deciles, 0-indexed, descending.
    Label 0 = highest return decile, label 9 = lowest.

    Uses rank(method="first") so ties are broken deterministically — this ALWAYS
    yields all 10 classes when len(fwd) >= 10, unlike qcut(duplicates="drop")
    which silently produces fewer bins (and a missing class) when returns tie
    (e.g. many delisting/illiquid zeros).  XGBoost multi:softprob with num_class=10
    requires every class to appear, so this robustness matters.
    """
    fwd = fwd.dropna()
    r   = fwd.rank(method="first")                       # 1..n ascending
    asc = (((r - 1) * 10 // len(r)).clip(upper=9)).astype(int)   # 0=lowest … 9=highest
    return (9 - asc).astype(int)                         # 0=highest return


def make_labels(rets: pd.DataFrame, t: int) -> pd.Series:
    """Next-month cross-sectional return deciles over the full cross-section."""
    return _decile_labels(rets.iloc[t].dropna())


# ── 4. XGBoost ensemble ───────────────────────────────────────────────────────
def train_ensemble(
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_va: pd.DataFrame, y_va: pd.Series,
) -> list:
    """Train N_SEEDS XGBClassifiers with different random seeds."""
    models = []
    for seed in range(N_SEEDS):
        m = XGBClassifier(**XGB_PARAMS, random_state=seed)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        models.append(m)
    return models


def predict_ensemble(models: list, X: pd.DataFrame) -> np.ndarray:
    """Average probability matrix over the ensemble;  shape (N_stocks, 10)."""
    return np.mean([m.predict_proba(X) for m in models], axis=0)


# ── 5. Reclassification scores ────────────────────────────────────────────────
def score_dpr(probs: np.ndarray) -> np.ndarray:
    """
    DPR — paper Eq. 21 (PrDf5).
    Linear-class-mean assumption: score = Σ_{k=1}^{5} (p_k - p_{11-k}) * (6-k)
    probs[:,0] = P(class 0) = P(top-return decile).
    """
    s = np.zeros(len(probs))
    for k in range(1, 6):
        s += (probs[:, k - 1] - probs[:, 10 - k]) * (6 - k)
    return s


def score_ret(probs: np.ndarray, mu_k: np.ndarray) -> np.ndarray:
    """RET — paper Eq. 22.  Expected return: μ_i = Σ_k p_{i,k} * μ_k."""
    return probs @ mu_k


def score_srp(
    probs: np.ndarray, mu_k: np.ndarray, sigma2_k: np.ndarray
) -> np.ndarray:
    """
    SRP — paper Eqs. 23–24.
    Predicted Sharpe via law of total variance:
        μ_i   = probs @ mu_k
        Var_i = probs @ (σ²_k + μ²_k) - μ²_i
        SRP_i = μ_i / sqrt(Var_i)
    """
    mu_i  = probs @ mu_k
    var_i = probs @ (sigma2_k + mu_k ** 2) - mu_i ** 2
    return mu_i / np.sqrt(np.maximum(var_i, 1e-12))


# ── 6. Portfolio construction & metrics ───────────────────────────────────────
def port_ret(
    scores: np.ndarray, idx: pd.Index, fwd: pd.Series, q: float = TOP_Q
) -> tuple:
    """Equal-weighted long top-q / short bottom-q.  Returns (ls_ret, long_ret)."""
    s   = pd.Series(scores, index=idx)
    r   = fwd.reindex(s.index).dropna()
    s   = s.reindex(r.index)
    n   = max(1, int(len(s) * q))
    lo  = r[s.nlargest(n).index].mean()
    sh  = r[s.nsmallest(n).index].mean()
    return lo - sh, lo


def perf(series: list) -> tuple:
    """Returns (annualised_return, Sharpe, max_drawdown)."""
    r   = np.array(series)
    ann = r.mean() * 12
    vol = r.std() * np.sqrt(12)
    sr  = ann / vol if vol > 0 else np.nan
    cum = np.cumprod(1 + r)
    mdd = (cum / np.maximum.accumulate(cum) - 1).min()
    return ann, sr, mdd


# ── 7. Main backtest loop ─────────────────────────────────────────────────────
def backtest(csv_path=None):
    rets, size = load_data(csv_path)
    T = len(rets)
    print(
        f"Data: {T} months × {rets.shape[1]} stocks  "
        f"({rets.index[0].date()} – {rets.index[-1].date()})"
    )

    first_feat = max(MOM_WINDOWS) + 1          # need 12 months of history for features
    first_pred = MIN_TRAIN_YRS * 12 + first_feat

    if first_pred >= T:
        raise ValueError(f"Need at least {first_pred} months of data; got {T}.")

    # Pre-compute features and labels once for all t ∈ [first_feat, T)
    # At time t:  features use data through t-1,  label = return decile at t.
    # Portfolio formed at end of t-1, held during t  →  fwd return = rets.iloc[t].
    print("Pre-computing features / labels...")
    pool = {}   # t → (features DataFrame, labels Series)
    for t in range(first_feat, T):
        F   = make_features(rets, size, t).dropna()
        idx = F.index.intersection(rets.iloc[t].dropna().index)
        if len(idx) >= 20:
            L = _decile_labels(rets.iloc[t].reindex(idx))   # deciles within trained set
            pool[t] = (F.loc[idx], L)

    results     = {k: {"ls": [], "lo": []} for k in ("bench", "dpr", "ret", "srp")}
    model_store = {}   # year → (models, mu_k, sigma2_k)

    pred_years = sorted({rets.index[t].year for t in range(first_pred, T)})

    for year in pred_years:
        months = [t for t in range(first_pred, T) if rets.index[t].year == year]
        if not months:
            continue
        t_cut = months[0]

        # ── Annual retraining, expanding window — time-blocked split (paper §4.1) ─
        all_ts_b = sorted([t for t in pool if t < t_cut])
        n_val_b  = max(12, int(len(all_ts_b) * 0.2))
        n_tr_b   = len(all_ts_b) - n_val_b
        tr_ts    = all_ts_b[:n_tr_b]      # earliest 80% of months
        va_ts    = all_ts_b[n_tr_b:]      # most recent 20% held out for early stopping

        if len(tr_ts) >= 24 and len(va_ts) >= 12:
            X_tr = pd.concat([pool[t][0] for t in tr_ts])
            y_tr = pd.concat([pool[t][1] for t in tr_ts])
            X_va = pd.concat([pool[t][0] for t in va_ts])
            y_va = pd.concat([pool[t][1] for t in va_ts])

            # Estimate decile mean / variance on training returns (for RET / SRP)
            fwd_tr = pd.concat(
                [rets.iloc[t].reindex(pool[t][0].index) for t in tr_ts]
            )
            lab_tr = pd.concat([pool[t][1] for t in tr_ts])
            grp    = (
                pd.DataFrame({"l": lab_tr.values, "r": fwd_tr.values})
                .groupby("l")["r"]
            )
            mu_k    = np.array([
                grp.get_group(k).mean() if k in grp.groups else 0.0
                for k in range(10)
            ])
            sigma2_k = np.array([
                grp.get_group(k).var() if k in grp.groups else 1e-4
                for k in range(10)
            ])

            print(f"  {year}: training  train={len(X_tr):,}  val={len(X_va):,} obs")
            models = train_ensemble(X_tr, y_tr, X_va, y_va)
            model_store[year] = (models, mu_k, sigma2_k)

        elif model_store:
            # Reuse most recent model if not enough data yet
            model_store[year] = list(model_store.values())[-1]
        else:
            continue

        mdls, mu_k, sigma2_k = model_store[year]

        # ── Monthly prediction & portfolio formation ───────────────────────
        for t in months:
            if t not in pool:
                continue
            F, _  = pool[t]
            fwd   = rets.iloc[t].reindex(F.index).dropna()   # return during month t
            if len(fwd) < 10:
                continue
            F     = F.reindex(fwd.index)
            probs = predict_ensemble(mdls, F)

            # Benchmark: naive cross-sectional zMOM12
            bl, blo = port_ret(F["zMOM12"].values, F.index, fwd)
            results["bench"]["ls"].append(bl)
            results["bench"]["lo"].append(blo)

            for name, sc in [
                ("dpr", score_dpr(probs)),
                ("ret", score_ret(probs, mu_k)),
                ("srp", score_srp(probs, mu_k, sigma2_k)),
            ]:
                ls, lo = port_ret(sc, F.index, fwd)
                results[name]["ls"].append(ls)
                results[name]["lo"].append(lo)

    # ── Performance report ────────────────────────────────────────────────────
    n_months = len(results["bench"]["ls"])
    print(f"\n── Performance ({n_months} prediction months) " + "─" * 24)
    print(f"{'Strategy':<26} {'Ann.Ret':>9} {'Sharpe':>9} {'MaxDD':>9}")
    print("─" * 57)
    for name, label in [
        ("bench", "Bench zMOM12"),
        ("dpr",   "DM-DPR"),
        ("ret",   "DM-RET"),
        ("srp",   "DM-SRP"),
    ]:
        for key, tag in (("ls", "L/S"), ("lo", "Long")):
            ann, sr, mdd = perf(results[name][key])
            print(f"{label + ' ' + tag:<26} {ann:>9.2%} {sr:>9.2f} {mdd:>9.2%}")
        print()

    return results


def compute_eligibility(
    prices_monthly: pd.DataFrame,
    size_monthly: pd.DataFrame | None = None,
    min_price: float = 5.0,
    min_coverage: float = 0.70,
    window: int = 36,
    min_history: int = 12,
    min_dollar_vol_pct: float = 0.0,
    min_dollar_vol_abs: float = 5e6,
    short_min_dollar_vol_abs: float = 25e6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Point-in-time (causal) tradeability masks — NO look-ahead.  Returns
    (eligible, shortable), both T×N booleans aligned to prices_monthly.

    A stock is ELIGIBLE (long-tradeable) at month t iff, using ONLY data ≤ t:
      1. current price > min_price (penny-stock floor), AND
      2. price > min_price in ≥ min_coverage of the trailing window months
         (requires ≥ min_history valid months), AND
      3. (if size_monthly given) trailing 3-month avg dollar volume is in the top
         (1 − min_dollar_vol_pct) percentile AND above min_dollar_vol_abs.

    A stock is SHORTABLE iff it is eligible AND its trailing dollar volume clears
    a higher floor (short_min_dollar_vol_abs).  This is the borrowability / EU-SSR
    "locate" proxy: hard-to-borrow micro-caps cannot be legally / practically shorted,
    so they are removed from the short book entirely (borrow cost is applied to the
    rest via BACKTEST.tiered_borrow_fees).

    All rules use strictly backward rolling windows → no look-ahead.  Evaluate at the
    signal date (t−1) to get the point-in-time universe.
    """
    valid   = prices_monthly.notna()
    good_px = (prices_monthly > min_price) & valid
    n_valid = valid.rolling(window, min_periods=min_history).sum()
    n_good  = good_px.rolling(window, min_periods=min_history).sum()
    cov     = n_good / n_valid.clip(lower=1)
    elig    = (cov >= min_coverage) & (prices_monthly > min_price) & (n_valid >= min_history)
    shortable = elig.copy()

    if size_monthly is not None:
        # 3-month trailing average dollar volume (point-in-time, causal)
        dv = size_monthly.reindex_like(prices_monthly).rolling(3, min_periods=1).mean()
        if min_dollar_vol_pct > 0:
            elig = elig & (dv.rank(axis=1, pct=True) >= min_dollar_vol_pct)
        if min_dollar_vol_abs > 0:
            elig = elig & (dv >= min_dollar_vol_abs)
        # borrowability: stricter liquidity floor for the short side
        shortable = elig & (dv >= short_min_dollar_vol_abs)

    return elig.fillna(False), shortable.fillna(False)


def _build_pnl_prices(
    prices_monthly: pd.DataFrame,
    lo: float = -0.95,
    hi: float = 3.0,
) -> pd.DataFrame:
    """
    Price series for PnL with realised returns cleaned the same way the modelling
    returns are (Han §4.1): per-cell clip to [lo, hi] (kills split/feed data errors
    on the upside, keeps real crashes on the downside) then cross-sectional 1/99
    winsorisation each month.  A synthetic price path is rebuilt from the cleaned
    returns (levels are arbitrary; only ratios matter for PnL) and the original
    NaN structure is restored so the traded universe / features are unaffected.

    Without this, routing PnL through raw prices lets a single untradeable penny
    pop (+1000%/month) detonate the short side — the realistic universe filter
    handles most of it, this caps the residual data-error tail.
    """
    r    = prices_monthly.pct_change().clip(lower=lo, upper=hi)
    qlo  = r.quantile(0.01, axis=1)
    qhi  = r.quantile(0.99, axis=1)
    r    = r.clip(qlo, qhi, axis=0)
    synth = (1.0 + r.fillna(0.0)).cumprod()
    return synth.where(prices_monthly.notna())


def _inject_delisting_returns(
    prices_monthly: pd.DataFrame,
    delist_return: float = -0.30,
) -> pd.DataFrame:
    """
    Make delistings cost money.

    The backtest core treats a NaN return as 0, so a held name that simply
    disappears from the price matrix books a 0% return instead of its loss —
    silently amputating crash/short-squeeze risk (the same flaw the old ±50%
    clip had).  Here, for every column whose last valid price is before the end
    of the sample, we write ONE synthetic price at the following month equal to
    last_price * (1 + delist_return).  pct_change then realises the delisting
    return in the month after the last trade; the position is closed at the next
    rebalance (price is NaN thereafter).

    -30% is the Beaver-McNichols-Price (2007) / Han (2022 §4.1) fallback used
    when a CRSP delist return is unavailable.  The index is unchanged (we only
    fill an existing, previously-NaN cell), so weight/price date alignment holds.
    """
    px   = prices_monthly.copy()
    idx  = px.index
    last = len(idx) - 1
    for col in px.columns:
        s  = px[col]
        lv = s.last_valid_index()
        if lv is None:
            continue
        pos = idx.get_loc(lv)
        if pos < last:                       # stopped trading before sample end → delisted
            px.iloc[pos + 1, px.columns.get_loc(col)] = s.iloc[pos] * (1.0 + delist_return)
    return px


def fetch_tiingo_monthly(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str = "102cb09d2f83b832d38f00437fd18de26e025d95",
) -> pd.DataFrame:
    """
    Fetch a single ticker's monthly adjusted-close series from Tiingo.

    Returns a one-column DataFrame named `ticker`, month-end indexed.  Used for the
    SPY buy-and-hold benchmark so the whole pipeline depends only on Tiingo (no
    yfinance).  Raises on network/HTTP failure so a silent empty benchmark can't
    slip into the comparison.
    """
    import requests
    url = (
        f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        f"?startDate={start_date}&endDate={end_date}"
        f"&resampleFreq=monthly&token={api_key}"
    )
    resp = requests.get(
        url,
        headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError(f"Tiingo returned no data for {ticker} ({start_date}–{end_date}).")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    out = df.set_index("date")[["adjClose"]].rename(columns={"adjClose": ticker})
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def load_broad_universe_tiingo(
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    min_coverage: float = 0.70,
    min_price: float = 1.0,
    api_key: str = "102cb09d2f83b832d38f00437fd18de26e025d95",
    checkpoint_path: str = "tiingo_download_checkpoint.parquet",
    max_workers: int = 20,
    skip_download: bool = False,
    verbose: bool = True,
) -> tuple:
    """
    Download a survivorship-bias-free broad US equity universe from Tiingo.

    Fetches every NYSE/NASDAQ common stock active at any point since
    `start_date` (including delisted companies).  Monthly adj-close prices
    are downloaded via Tiingo's resampleFreq=monthly endpoint.

    Results are cached to `cache_path`; partial downloads are checkpointed
    to `checkpoint_path` so interrupted runs resume automatically.

    Returns
    -------
    prices_monthly : pd.DataFrame  monthly adj-close  (T × N)
    rets_monthly   : pd.DataFrame  monthly returns ±50% (T × N)
    size_monthly   : pd.DataFrame  adjClose × adjVolume proxy (T × N)
    """
    import os, io, zipfile, requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    # ── 1. Checkpoint ─────────────────────────────────────────────────────────

    # ── 2. Load checkpoint ────────────────────────────────────────────────────
    done: dict[str, pd.DataFrame] = {}
    if os.path.exists(checkpoint_path):
        try:
            ckpt = pd.read_parquet(checkpoint_path)
            if isinstance(ckpt.columns, pd.MultiIndex):
                for ticker in ckpt["close"].columns:
                    sub = pd.DataFrame({
                        "adjClose":  ckpt["close"][ticker],
                        "adjVolume": ckpt["volume"][ticker],
                    }).dropna(how="all")
                    if not sub.empty:
                        done[ticker] = sub
                if verbose:
                    print(f"[tiingo] checkpoint: {len(done):,} tickers loaded")
        except Exception:
            pass

    if skip_download:
        if not done:
            raise ValueError("skip_download=True but no checkpoint found at " + checkpoint_path)
        if verbose:
            print(f"[tiingo] skipping download — using {len(done):,} tickers from checkpoint")
    else:
        # ── 3. Get full ticker universe and download remaining ────────────────
        TICKER_ZIP_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
        if verbose:
            print("[tiingo] downloading ticker universe …")
        r = requests.get(TICKER_ZIP_URL, timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            ticker_df = pd.read_csv(z.open(z.namelist()[0]))

        us = ticker_df[
            ticker_df["exchange"].isin(["NYSE", "NASDAQ"]) &
            (ticker_df["assetType"] == "Stock")
        ].copy()
        us["startDate"] = pd.to_datetime(us["startDate"], errors="coerce")
        us["endDate"]   = pd.to_datetime(us["endDate"],   errors="coerce")
        us = us[us["endDate"].isna() | (us["endDate"] >= pd.Timestamp(start_date))]
        tickers = us["ticker"].dropna().unique().tolist()

        if verbose:
            print(f"[tiingo] {len(tickers):,} US stocks  |  {len(done):,} already in checkpoint")

        remaining = [t for t in tickers if t not in done]
        if verbose:
            print(f"[tiingo] {len(remaining):,} tickers left to fetch …")

        # ── 4. Download in parallel with rate-limit retry ────────────────────
        import threading, time as _time
        BASE    = "https://api.tiingo.com/tiingo/daily"
        HEADERS = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
        _rate_lock  = threading.Lock()
        _last_429   = [0.0]

        def fetch_ticker(ticker: str) -> tuple[str, pd.DataFrame | None]:
            url = (
                f"{BASE}/{ticker}/prices"
                f"?startDate={start_date}&endDate={end_date}"
                f"&resampleFreq=monthly&token={api_key}"
            )
            for attempt in range(5):
                with _rate_lock:
                    wait = _last_429[0] + 65 - _time.time()
                if wait > 0:
                    _time.sleep(wait)
                try:
                    resp = requests.get(url, headers=HEADERS, timeout=20)
                    if resp.status_code == 404:
                        return ticker, None
                    if resp.status_code == 429:
                        with _rate_lock:
                            _last_429[0] = _time.time()
                        _time.sleep(65 + attempt * 10)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    if not data:
                        return ticker, None
                    df = pd.DataFrame(data)
                    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                    df = df.set_index("date")[["adjClose", "adjVolume"]].dropna(how="all")
                    return ticker, df
                except Exception:
                    _time.sleep(2 ** attempt)
            return ticker, None

        CHECKPOINT_EVERY = 200
        fetched_since_ckpt = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_ticker, t): t for t in remaining}
            n_total = len(futures)
            n_done  = 0
            for fut in as_completed(futures):
                ticker, df = fut.result()
                n_done += 1
                if df is not None and not df.empty:
                    done[ticker] = df
                fetched_since_ckpt += 1

                if verbose and n_done % 500 == 0:
                    print(f"[tiingo] {n_done:,}/{n_total:,} fetched  "
                          f"({len(done):,} with data)")

                if fetched_since_ckpt >= CHECKPOINT_EVERY:
                    _save_tiingo_checkpoint(done, checkpoint_path)
                    fetched_since_ckpt = 0

        _save_tiingo_checkpoint(done, checkpoint_path)
        if verbose:
            print(f"[tiingo] download complete: {len(done):,} tickers with data")

    # ── 5. Build wide matrices ─────────────────────────────────────────────────
    close_wide  = pd.DataFrame({t: done[t]["adjClose"]  for t in done}).sort_index()
    volume_wide = pd.DataFrame({t: done[t]["adjVolume"] for t in done}).sort_index()

    close_wide.index  = pd.to_datetime(close_wide.index)
    volume_wide.index = pd.to_datetime(volume_wide.index)

    # ── Universe filter — POINT-IN-TIME, no look-ahead ───────────────────────
    # The old filter kept a stock if price>min in ≥X% of *all* its months — that
    # uses future prices to decide today's membership (look-ahead/survivorship).
    # Here we only drop columns that can never form a feature (<12 valid months
    # anywhere).  The real, time-varying coverage/price test is applied causally
    # at portfolio-formation time via compute_eligibility().  Delisted names are
    # kept → survivorship-bias-free.
    n_valid_ever = close_wide.notna().sum()
    keep         = n_valid_ever >= 12
    close_wide   = close_wide.loc[:, keep]
    volume_wide  = volume_wide.reindex(columns=close_wide.columns)
    if verbose:
        print(f"[tiingo] kept {int(keep.sum()):,}/{len(keep):,} tickers "
              f"(≥12 valid months ever; point-in-time coverage applied at trade time)")

    prices_raw     = close_wide
    volume_monthly = volume_wide

    # Inject explicit delisting returns FIRST so rets_monthly and pnl_prices
    # share the same source.  A name whose last real price is before sample end
    # gets a synthetic price at pos+1 = last_price * 0.70 (−30% Han §4.1 fallback).
    # Using this as the base for rets_monthly means training labels SEE the −30%
    # and learn it → bottom decile; PnL books the same loss.  Consistent.
    prices_monthly = _inject_delisting_returns(prices_raw, delist_return=-0.30)

    # Returns for features / labels / decile means.  >+300% = upside data error → NaN.
    # Downside kept; cross-sectional 1/99 winsorization each month.
    _raw         = prices_monthly.pct_change()
    _raw         = _raw.mask(_raw > 3.0)
    _lo          = _raw.quantile(0.01, axis=1)
    _hi          = _raw.quantile(0.99, axis=1)
    rets_monthly = _raw.clip(_lo, _hi, axis=0)

    size_monthly = (prices_raw * volume_monthly).ffill()

    if verbose:
        print(f"[tiingo] monthly prices : {prices_monthly.shape}  "
              f"({prices_monthly.index[0].date()} – {prices_monthly.index[-1].date()})")

    return prices_monthly, rets_monthly, size_monthly


def _save_tiingo_checkpoint(done: dict, checkpoint_path: str) -> None:
    """Atomically write checkpoint: write to .tmp then rename so a crash never corrupts."""
    import os
    tmp = checkpoint_path + ".tmp"
    try:
        close_df  = pd.DataFrame({t: done[t]["adjClose"]  for t in done})
        volume_df = pd.DataFrame({t: done[t]["adjVolume"] for t in done})
        pd.concat({"close": close_df, "volume": volume_df}, axis=1).to_parquet(tmp)
        os.replace(tmp, checkpoint_path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ── Archive data loader & BACKTEST.py integration ────────────────────────────

def load_sp500_archive(csv_path: str) -> tuple:
    """
    Load the Kaggle S&P 500 all_stocks_5yr.csv archive (or any file with the
    same columns: date, open, high, low, close, volume, Name).

    Returns
    -------
    prices_daily  : pd.DataFrame  daily close prices      (T_daily  × N_stocks)
    rets_monthly  : pd.DataFrame  monthly returns ±50%    (T_monthly × N_stocks)
    size_monthly  : pd.DataFrame  Close×Volume proxy       (T_monthly × N_stocks)
    """
    df = (
        pd.read_csv(csv_path, parse_dates=["date"])
        .sort_values("date")
    )
    df["date"] = pd.to_datetime(df["date"])

    # Wide daily close prices — required by BACKTEST.backtest()
    prices_daily = (
        df.pivot_table(index="date", columns="Name", values="close", aggfunc="last")
        .sort_index()
    )
    prices_daily.index = pd.to_datetime(prices_daily.index)
    prices_daily.columns.name = None

    # Monthly (month-end) prices and volume for the DM feature engine
    prices_monthly = prices_daily.resample("ME").last()

    volume_daily = (
        df.pivot_table(index="date", columns="Name", values="volume", aggfunc="last")
        .sort_index()
    )
    volume_daily.index = pd.to_datetime(volume_daily.index)
    volume_monthly = volume_daily.resample("ME").last()

    size_monthly  = prices_monthly * volume_monthly
    rets_monthly  = prices_monthly.pct_change().clip(-0.5, 0.5)

    print(
        f"[load] daily prices : {prices_daily.shape}  "
        f"({prices_daily.index[0].date()} – {prices_daily.index[-1].date()})"
    )
    print(
        f"[load] monthly rets : {rets_monthly.shape}  "
        f"({rets_monthly.index[0].date()} – {rets_monthly.index[-1].date()})"
    )
    return prices_daily, rets_monthly, size_monthly


def generate_dm_weights(
    rets: pd.DataFrame,
    size: pd.DataFrame,
    *,
    strategy: str = "srp",
    portfolio: str = "ls",
    q: float = TOP_Q,
    min_train_months: int = 18,
    n_seeds: int = N_SEEDS,
) -> pd.DataFrame:
    """
    Run the Deep Momentum signal loop and return a monthly weights DataFrame
    ready to plug into BACKTEST.py's ``backtest()`` function.

    Parameters
    ----------
    rets, size      : monthly DataFrames produced by ``load_sp500_archive()``
    strategy        : ``'dpr'``, ``'ret'``, ``'srp'``, or ``'bench'`` (zMOM12)
    portfolio       : ``'ls'`` (long-short, equal-weight ±1/n) or ``'lo'`` (long-only)
    min_train_months: minimum months of history before first prediction.
                      Use ≤30 for the 5-year archive; the paper uses 120 (10 years).
    n_seeds         : XGBoost ensemble size (reduce for faster runs, e.g. 3)

    Returns
    -------
    weights_df : pd.DataFrame  indexed by signal dates (month-end of the period
                               *before* the holding month), columns = tickers.
                               Designed for forward-fill into a daily prices index.

    Timing convention
    -----------------
    Signal date  = rets.index[t-1]  (last trading day of month t-1)
    Hold period  = month t          (rets.iloc[t])
    With lag=1 in BACKTEST.backtest(), execution lands on the first
    trading day of month t — no look-ahead.
    """
    T          = len(rets)
    first_feat = max(MOM_WINDOWS) + 1
    first_pred = min_train_months + first_feat
    if first_pred >= T - 1:
        raise ValueError(
            f"Need ≥ {first_pred + 1} monthly observations; got {T}. "
            f"Reduce min_train_months (currently {min_train_months})."
        )

    # Pre-compute features and labels once (no FFD in this single-strategy path)
    pool = {}
    for t in range(first_feat, T - 1):
        F   = make_features(rets, size, t, ffd_scores=None).dropna()
        idx = F.index.intersection(rets.iloc[t].dropna().index)
        if len(idx) >= 20:
            L = _decile_labels(rets.iloc[t].reindex(idx))   # deciles within trained set
            pool[t] = (F.loc[idx], L)

    weight_rows = {}   # signal_date → pd.Series of weights
    model_store = {}   # year → (models, mu_k, sigma2_k)

    pred_years = sorted({rets.index[t].year for t in range(first_pred, T - 1)})

    for year in pred_years:
        months = [t for t in range(first_pred, T - 1) if rets.index[t].year == year]
        if not months:
            continue
        t_cut  = months[0]
        all_ts = sorted([t for t in pool if t < t_cut])
        n_val  = max(6, int(len(all_ts) * 0.2))

        if len(all_ts) >= 18:
            fwd_all  = pd.concat([rets.iloc[t].reindex(pool[t][0].index) for t in all_ts])
            lab_all  = pd.concat([pool[t][1] for t in all_ts])
            grp      = pd.DataFrame({"l": lab_all.values, "r": fwd_all.values}).groupby("l")["r"]
            mu_k     = np.array([grp.get_group(k).mean() if k in grp.groups else 0.0  for k in range(10)])
            sigma2_k = np.array([grp.get_group(k).var()  if k in grp.groups else 1e-4 for k in range(10)])

            # Time-blocked split: last 20% of months as validation (paper Sec. 3.3.3)
            n_tr    = len(all_ts) - n_val
            tr_ts_s = all_ts[:n_tr]
            va_ts_s = all_ts[n_tr:]
            X_tr_s  = pd.concat([pool[t][0] for t in tr_ts_s if t in pool])
            y_tr_s  = pd.concat([pool[t][1] for t in tr_ts_s if t in pool])
            X_va_s  = pd.concat([pool[t][0] for t in va_ts_s if t in pool])
            y_va_s  = pd.concat([pool[t][1] for t in va_ts_s if t in pool])
            print(f"  [{year}] pool={len(all_ts)}  n_val={n_val}  seeds={n_seeds}")
            models = []
            for seed in range(n_seeds):
                m = XGBClassifier(**XGB_PARAMS, random_state=seed).fit(
                    X_tr_s, y_tr_s, eval_set=[(X_va_s, y_va_s)], verbose=False
                )
                models.append(m)
            model_store[year] = (models, mu_k, sigma2_k)

        elif model_store:
            model_store[year] = list(model_store.values())[-1]
        else:
            continue

        mdls, mu_k, sigma2_k = model_store[year]

        for t in months:
            if t not in pool:
                continue
            F, _         = pool[t]
            signal_date  = rets.index[t - 1]   # end of month t-1 → trade at open of month t

            probs = np.mean([m.predict_proba(F) for m in mdls], axis=0)

            if strategy == "bench":
                scores = F["zMOM12"].values
            elif strategy == "dpr":
                scores = score_dpr(probs)
            elif strategy == "ret":
                scores = score_ret(probs, mu_k)
            elif strategy == "srp":
                scores = score_srp(probs, mu_k, sigma2_k)
            else:
                raise ValueError(f"Unknown strategy '{strategy}'. Use 'dpr','ret','srp','bench'.")

            s_ser = pd.Series(scores, index=F.index)
            n     = max(1, int(len(s_ser) * q))
            w     = pd.Series(0.0, index=F.index)
            w[s_ser.nlargest(n).index]  = +1.0 / n
            if portfolio == "ls":
                w[s_ser.nsmallest(n).index] = -1.0 / n

            weight_rows[signal_date] = w

    if not weight_rows:
        raise ValueError(
            "No weights generated. Increase data length or decrease min_train_months."
        )

    weights_df = (
        pd.DataFrame(weight_rows)
        .T
        .fillna(0.0)
        .sort_index()
    )
    weights_df.index.name = "date"
    print(
        f"[weights] {len(weights_df)} signal dates  "
        f"({weights_df.index[0].date()} – {weights_df.index[-1].date()})"
    )
    return weights_df


def run_with_backtest(
    csv_path: str,
    *,
    strategy: str = "srp",
    portfolio: str = "ls",
    q: float = TOP_Q,
    min_train_months: int = 18,
    transaction_cost: float = 0.001,
    freq: int = 252,
    lag: int = 1,
    n_seeds: int = 3,
) -> dict:
    """
    Full pipeline for the Kaggle S&P 500 archive:
        load data  →  generate DM weights  →  call BACKTEST.backtest()

    Requires ``BACKTEST.py`` in the same directory (or on ``sys.path``).

    Parameters
    ----------
    csv_path         : path to ``all_stocks_5yr.csv`` (columns: date,open,high,low,close,volume,Name)
    strategy         : ``'dpr'``, ``'ret'``, ``'srp'``, or ``'bench'``
    portfolio        : ``'ls'`` (long-short) or ``'lo'`` (long-only)
    min_train_months : keep ≤ 30 for the 5-year archive; default 18 gives ~3 years OOS
    transaction_cost : round-trip cost as fraction of traded notional (e.g. 0.001 = 10 bps)
    freq             : annualisation factor (252 for daily prices)
    lag              : execution lag in trading days (1 = signal at close t → trade at close t+1)
    n_seeds          : ensemble size; reduce to 3–5 for faster testing

    Returns
    -------
    dict from BACKTEST.backtest() with keys:
        returns, equity, drawdown, ann_return, sharpe, max_drawdown, …
    """
    try:
        import BACKTEST
    except ImportError:
        raise ImportError(
            "BACKTEST.py not found. Make sure BACKTEST.py is in the same directory."
        )

    # 1. Load data
    prices_daily, rets_monthly, size_monthly = load_sp500_archive(csv_path)

    # 2. Generate DM weights (monthly signal dates)
    print(f"\nGenerating DM-{strategy.upper()} {portfolio.upper()} weights …")
    weights = generate_dm_weights(
        rets_monthly, size_monthly,
        strategy=strategy, portfolio=portfolio,
        q=q, min_train_months=min_train_months,
        n_seeds=n_seeds,
    )

    # 3. Map each monthly signal date to the nearest prior trading day in prices_daily
    #    (month-end dates from resample may be weekends or holidays)
    daily_idx   = prices_daily.index
    signal_dates = []
    mapped_index = []
    for d in weights.index:
        prior = daily_idx[daily_idx <= d]
        if len(prior):
            td = prior[-1]
            signal_dates.append(td)
            mapped_index.append(td)

    weights_mapped = weights.copy()
    weights_mapped.index = pd.DatetimeIndex(mapped_index)
    weights_mapped = weights_mapped[~weights_mapped.index.duplicated(keep="last")]

    # 4. Restrict daily prices to tickers present in weights and trim to OOS start
    common_tickers = weights_mapped.columns.intersection(prices_daily.columns).tolist()
    oos_start      = signal_dates[0]
    prices_oos     = prices_daily.loc[oos_start:, common_tickers]
    weights_oos    = weights_mapped.reindex(columns=common_tickers).fillna(0.0)

    print(
        f"\n[backtest] OOS window : {oos_start.date()} – {prices_oos.index[-1].date()}"
        f"  ({len(prices_oos)} trading days,  {len(signal_dates)} rebalances)"
    )
    print(f"[backtest] Universe   : {len(common_tickers)} tickers")

    # 5. Call BACKTEST.backtest()
    result = BACKTEST.backtest(
        weights=weights_oos,
        prices=prices_oos,
        freq=freq,
        lag=lag,
        transaction_cost=transaction_cost,
        signal_dates=signal_dates,
    )

    # 6. Print summary
    print(f"\n── DM-{strategy.upper()} {portfolio.upper()} Results {'─'*30}")
    print(f"  Annual Return  : {result['ann_return']:>8.2%}")
    print(f"  Annual Vol     : {result['ann_vol']:>8.2%}")
    print(f"  Sharpe Ratio   : {result['sharpe']:>8.3f}")
    print(f"  Max Drawdown   : {result['max_drawdown']:>8.2%}")
    print(f"  Total Return   : {result['total_return']:>8.2%}")
    print(f"  Ann. Turnover  : {result['ann_turnover']:>8.2%}")

    return result


# ── Multi-strategy comparison helpers ────────────────────────────────────────

def _ffd_from_training_window(prices_monthly: pd.DataFrame, t_cut: int) -> dict:
    """
    Compute FFD scores with NO look-ahead:
      1. Find optimal d using prices up to t_cut (training data only).
      2. Apply the FFD filter causally to the full price series (lfilter is causal
         — y[t] depends only on x[0..t] — so using future rows is safe).

    This is the correct way to use FFD inside a walk-forward backtest.
    """
    from ffd import find_optimal_d_batch, build_ffd_scores_v2
    prices_train = prices_monthly.iloc[:t_cut]
    d_series = find_optimal_d_batch(prices_train, n_jobs=-1, verbose=False)
    # v2: uniform median d (cross-sectionally coherent) + FFD level & ΔFFD slopes.
    # Causal filter applied to full series → no leakage beyond (training-only) d selection.
    return build_ffd_scores_v2(prices_monthly, d_series, windows=[1, 3, 12])


def _bench_weights(
    rets: pd.DataFrame,
    size: pd.DataFrame,
    prices_monthly: pd.DataFrame,
    *,
    portfolio: str = "ls",
    q: float = TOP_Q,
    min_train_months: int = MIN_TRAIN_YRS * 12,
    use_ffd: bool = True,
    ffd_scores: dict | None = None,
    eligible: pd.DataFrame | None = None,
    shortable: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Cross-sectional momentum long(/short) — no XGBoost.
    ffd_scores: pre-computed FFD scores (skips re-estimation if provided).
    eligible  : point-in-time tradeability mask (compute_eligibility); applied at
                the signal date t-1 so the traded universe has no look-ahead.
    shortable : borrowability mask; the short leg is restricted to these names.
    """
    T          = len(rets)
    first_feat = max(MOM_WINDOWS) + 1
    first_pred = min_train_months + first_feat
    weight_rows = {}

    signal_col = "zMOM12"
    if use_ffd:
        if ffd_scores is None:
            print("  [bench FFD] computing optimal d on initial training window …")
            ffd_scores = _ffd_from_training_window(prices_monthly, first_pred)
            for m in list(ffd_scores.keys()):
                ffd_scores[m] = ffd_scores[m].reindex(rets.index)
        signal_col = "zFFD12"

    for t in range(first_pred, T - 1):
        F = make_features(rets, size, t, ffd_scores=ffd_scores).dropna()
        if eligible is not None:
            e = eligible.iloc[t - 1]
            F = F.loc[F.index.intersection(e.index[e.values])]
        if len(F) < 20:
            continue
        signal_date = rets.index[t - 1]
        s_ser = F[signal_col]
        n     = max(1, int(len(s_ser) * q))
        w     = pd.Series(0.0, index=F.index)
        w[s_ser.nlargest(n).index]  = +1.0 / n
        if portfolio == "ls":
            s_short = s_ser
            if shortable is not None:                       # borrowability: short only HTB-eligible
                sh = shortable.iloc[t - 1]
                s_short = s_ser.loc[s_ser.index.intersection(sh.index[sh.values])]
            w[s_short.nsmallest(n).index] = -1.0 / n
        weight_rows[signal_date] = w

    df = pd.DataFrame(weight_rows).T.fillna(0.0).sort_index()
    df.index.name = "date"
    print(f"[bench] {len(df)} signal dates  ({df.index[0].date()} – {df.index[-1].date()})")
    return df


def _generate_all_dm_weights(
    rets: pd.DataFrame,
    size: pd.DataFrame,
    prices_monthly: pd.DataFrame,
    *,
    portfolio: str = "ls",
    q: float = TOP_Q,
    min_train_months: int = MIN_TRAIN_YRS * 12,
    max_train_months: int | None = 120,
    n_seeds: int = N_SEEDS,
    use_ffd: bool = True,
    ffd_scores: dict | None = None,
    pool: dict | None = None,
    eligible: pd.DataFrame | None = None,
    shortable: pd.DataFrame | None = None,
) -> dict:
    """
    Train XGBoost (rolling window), then score DPR / RET / SRP.
    max_train_months: int = rolling cap (default 60); None = expanding window (paper §4.1).
    ffd_scores / pool: pass pre-computed values to skip recomputation.
    eligible: point-in-time tradeability mask (compute_eligibility), applied at the
              signal date t-1 to both the training pool and the traded universe.
    shortable: borrowability mask; the short leg is restricted to these names.
    """
    T          = len(rets)
    first_feat = max(MOM_WINDOWS) + 1
    first_pred = min_train_months + first_feat

    if first_pred >= T - 1:
        raise ValueError(f"Need ≥ {first_pred + 1} monthly obs; got {T}.")

    if use_ffd and ffd_scores is None:
        print("  [DM FFD] initial optimal-d search on first training window …")
        ffd_scores = _ffd_from_training_window(prices_monthly, first_pred)
        for m in list(ffd_scores.keys()):
            ffd_scores[m] = ffd_scores[m].reindex(rets.index)

    if pool is None:
        pool = {}
        for t in range(first_feat, T - 1):
            F   = make_features(rets, size, t, ffd_scores=ffd_scores).dropna()
            idx = F.index.intersection(rets.iloc[t].dropna().index)
            if eligible is not None:
                e   = eligible.iloc[t - 1]
                idx = idx.intersection(e.index[e.values])
            if len(idx) >= 20:
                # Deciles computed WITHIN the trained (eligible) cross-section → all
                # 10 classes present (a global decile rank could leave the subset
                # missing an extreme class and break XGBoost num_class=10).
                L = _decile_labels(rets.iloc[t].reindex(idx))
                pool[t] = (F.loc[idx], L)

    rows        = {"ret": {}}
    model_store = {}
    pred_years  = sorted({rets.index[t].year for t in range(first_pred, T - 1)})

    for year in pred_years:
        months = [t for t in range(first_pred, T - 1) if rets.index[t].year == year]
        if not months:
            continue
        t_cut  = months[0]
        all_ts = sorted([t for t in pool if t < t_cut])
        # Rolling window: keep only the most recent max_train_months
        if max_train_months and len(all_ts) > max_train_months:
            all_ts = all_ts[-max_train_months:]
        n_val  = max(6, int(len(all_ts) * 0.2))

        if len(all_ts) >= 18:
            # Decile mean returns μ_k for the RET reclassification (law of total expectation)
            fwd_all  = pd.concat([rets.iloc[t].reindex(pool[t][0].index) for t in all_ts if t in pool])
            lab_all  = pd.concat([pool[t][1] for t in all_ts if t in pool])
            grp      = pd.DataFrame({"l": lab_all.values, "r": fwd_all.values}).groupby("l")["r"]
            mu_k     = np.array([grp.get_group(k).mean() if k in grp.groups else 0.0 for k in range(10)])

            # Time-blocked split: last 20% of months as validation (paper Sec. 3.3.3)
            n_tr    = len(all_ts) - n_val
            tr_ts_s = all_ts[:n_tr]
            va_ts_s = all_ts[n_tr:]
            X_tr_s  = pd.concat([pool[t][0] for t in tr_ts_s if t in pool])
            y_tr_s  = pd.concat([pool[t][1] for t in tr_ts_s if t in pool])
            X_va_s  = pd.concat([pool[t][0] for t in va_ts_s if t in pool])
            y_va_s  = pd.concat([pool[t][1] for t in va_ts_s if t in pool])
            print(f"  [{year}] pool={len(all_ts)}  n_val={n_val}  seeds={n_seeds}")
            models = []
            for seed in range(n_seeds):
                m = XGBClassifier(**XGB_PARAMS, random_state=seed).fit(
                    X_tr_s, y_tr_s, eval_set=[(X_va_s, y_va_s)], verbose=False
                )
                models.append(m)
            model_store[year] = (models, mu_k)

        elif model_store:
            model_store[year] = list(model_store.values())[-1]
        else:
            continue

        mdls, mu_k = model_store[year]

        for t in months:
            if t not in pool:
                continue
            F, _ = pool[t]   # same cross-section used for training; no re-computation
            if len(F) < 20:
                continue
            signal_date = rets.index[t - 1]
            probs       = np.mean([m.predict_proba(F) for m in mdls], axis=0)

            sc    = score_ret(probs, mu_k)          # DM-RET reclassification (Σ pₖμₖ)
            s_ser = pd.Series(sc, index=F.index)
            n     = max(1, int(len(s_ser) * q))
            w     = pd.Series(0.0, index=F.index)
            w[s_ser.nlargest(n).index]  = +1.0 / n
            if portfolio == "ls":
                s_short = s_ser
                if shortable is not None:               # borrowability: short only HTB-eligible
                    sh = shortable.iloc[t - 1]
                    s_short = s_ser.loc[s_ser.index.intersection(sh.index[sh.values])]
                w[s_short.nsmallest(n).index] = -1.0 / n
            rows["ret"][signal_date] = w

    out = {}
    for name, weight_rows in rows.items():
        if not weight_rows:
            raise ValueError(f"No weights generated for {name}.")
        df = pd.DataFrame(weight_rows).T.fillna(0.0).sort_index()
        df.index.name = "date"
        print(f"[{name}] {len(df)} signal dates  ({df.index[0].date()} – {df.index[-1].date()})")
        out[name] = df

    return out


def _weights_to_daily(weights: pd.DataFrame, prices_daily: pd.DataFrame) -> tuple:
    """Map monthly weight index to nearest prior trading day.  Returns (mapped_df, signal_date_list)."""
    daily_idx    = prices_daily.index
    mapped_index = []
    for d in weights.index:
        prior = daily_idx[daily_idx <= d]
        if len(prior):
            mapped_index.append(prior[-1])
    w = weights.copy()
    w.index = pd.DatetimeIndex(mapped_index)
    w = w[~w.index.duplicated(keep="last")]
    return w, mapped_index


def apply_partial_adjustment(
    weights: pd.DataFrame, delta: float = 0.5, gross: float = 2.0
) -> pd.DataFrame:
    """
    Gârleanu–Pedersen partial adjustment (quadratic-cost-optimal turnover control).

    Optimal policy under quadratic (market-impact) costs is to trade only a
    fraction of the gap toward the target each period:
        w~_t = (1 - delta) * w~_{t-1} + delta * w*_t
    then renormalize the row to constant gross exposure (sum|w| = gross), which
    preserves dollar-neutrality (both inputs are net-zero → scaled combo is net-zero).

    delta in (0,1]:  1 = trade fully to target (= original strategy);
                     smaller delta = slower adjustment = lower turnover, mild alpha decay.
    The GP closed form sets delta from the cost/risk ratio:
        lambda*delta^2 + gamma*Sigma*delta - gamma*Sigma = 0.
    Here delta is exposed as a tunable knob (default 0.5 ≈ trade halfway each month).
    """
    cols   = weights.columns
    out    = {}
    w_prev = pd.Series(0.0, index=cols)
    for date, w_target in weights.iterrows():
        w = (1.0 - delta) * w_prev + delta * w_target
        s = w.abs().sum()
        if s > 0:
            w = w * (gross / s)          # hold gross constant; net-zero preserved
        out[date]  = w
        w_prev     = w
    return pd.DataFrame(out).T.reindex(columns=cols).fillna(0.0)


def compare_strategies(
    preloaded: tuple,
    *,
    n_seeds: int = N_SEEDS,
    min_train_months: int = MIN_TRAIN_YRS * 12,
    max_train_months: int | None = 120,
    q: float = TOP_Q,
    transaction_cost: float = 0.001,
    min_dollar_vol_pct: float = 0.0,
    min_dollar_vol_abs: float = 5e6,
    save_fig: str = "dm_comparison_tiingo.png",
) -> dict:
    """
    Run Bench (zMOM12 L/S), DM-RET, DM-GP L/S plus SPY B&H.
    preloaded          : (prices_monthly, rets_monthly, size_monthly) from load_broad_universe_tiingo()
    min_dollar_vol_pct : relative liquidity filter (0 = off, 0.7 = keep top 30%).
    min_dollar_vol_abs : absolute monthly dollar-volume floor (default $1M).
    """
    import matplotlib
    matplotlib.use("Agg")

    try:
        import BACKTEST
    except ImportError as e:
        raise ImportError(str(e))

    # ── 1. Unpack preloaded Tiingo data ───────────────────────────────────────
    prices_monthly, rets_monthly, size_monthly = preloaded
    data_start = rets_monthly.index[0]
    data_end   = rets_monthly.index[-1]

    # Point-in-time tradeability + borrowability masks (causal).  `shortable` is the
    # stricter (EU-SSR locate / hard-to-borrow) filter applied to the short book only.
    eligible, shortable = compute_eligibility(
        prices_monthly, size_monthly,
        min_dollar_vol_pct=min_dollar_vol_pct,
        min_dollar_vol_abs=min_dollar_vol_abs,
    )
    sel = rets_monthly.index
    print(f"[eligibility] avg tradeable names/month: "
          f"{eligible.loc[sel].sum(axis=1).iloc[min_train_months:].mean():.0f}  "
          f"| shortable: {shortable.loc[sel].sum(axis=1).iloc[min_train_months:].mean():.0f}")

    # Cleaned price series for PnL (winsorised realised returns + delisting),
    # so a single untradeable data-error move can't detonate the book.
    pnl_prices = _build_pnl_prices(prices_monthly)

    # Size/liquidity-tiered ONE-WAY transaction costs + ANNUAL short-borrow fees:
    # microcaps pay wide spreads and high borrow rates (size_monthly = dollar volume).
    tcost = BACKTEST.tiered_transaction_costs(size_monthly)
    bfee  = BACKTEST.tiered_borrow_fees(size_monthly)

    # ── 2. Bench weights (raw zMOM12 L/S) ────────────────────────────────────
    print("\n── Bench zMOM12 L/S ─────────────────────────────────────────────────")
    bench_w = _bench_weights(
        rets_monthly, size_monthly, prices_monthly,
        min_train_months=min_train_months, q=q, use_ffd=False,
        portfolio="ls", eligible=eligible, shortable=shortable,
    )

    # ── 3. DM weights (momentum + dynamics + FFD features) ───────────────────
    # use_ffd=True: optimal d frozen on the initial training window, causal filter
    # → FFD features added with no OOS look-ahead.
    print("\n── Deep Momentum L/S (+ FFD) ────────────────────────────────────────")
    dm_w = _generate_all_dm_weights(
        rets_monthly, size_monthly, prices_monthly,
        min_train_months=min_train_months, max_train_months=max_train_months,
        q=q, n_seeds=n_seeds, use_ffd=True,
        portfolio="ls", pool=None, eligible=eligible, shortable=shortable,
    )

    # ── 4. SPY benchmark (Tiingo — same source as the universe) ──────────────
    spy_prices = fetch_tiingo_monthly(
        "SPY",
        data_start.strftime("%Y-%m-%d"),
        data_end.strftime("%Y-%m-%d"),
    )

    # ── 5. Backtest each strategy THROUGH THE MAIN ENGINE (BACKTEST.py) ──────
    # Monthly bars + rebalance every month (signal_dates = every weight date) +
    # lag=0 → weights from signal at month m earn the return m→m+1 (the month
    # AFTER the signal: no look-ahead).  Rebalancing every period resets the book,
    # so the daily-drift gross-leverage blow-up that motivated the old standalone
    # backtester cannot occur — one engine, one timing convention.
    all_results = {}
    oos_start_ref = None

    def _run_engine(weights: pd.DataFrame, prices_m: pd.DataFrame, tc) -> dict:
        first = weights.index[0]
        px_bt = prices_m.loc[first:]                     # trim → correct annualisation
        sigs  = [d for d in weights.index if d in px_bt.index]
        w_bt  = weights.reindex(columns=px_bt.columns).fillna(0.0)
        return BACKTEST.backtest(
            w_bt, px_bt,
            freq=12, lag=0,
            transaction_cost=tc,
            borrow_fee=bfee,
            signal_dates=sigs,
        )

    #   DM       DM-RET reclassification (Σ pₖμₖ), equal-weight decile L/S
    #   DM-GP    DM + Gârleanu–Pedersen partial adjustment (delta=0.5)
    dm     = dm_w["ret"]
    dm_gp  = apply_partial_adjustment(dm, delta=0.5)
    for label, raw_w in [
        ("Bench zMOM12 L/S",  bench_w),   # raw momentum benchmark
        ("DM L/S",            dm),        # DM-RET reclassification
        ("DM-GP L/S",         dm_gp),     # DM + GP turnover control
    ]:
        oos_start = raw_w.index[0]
        if oos_start_ref is None:
            oos_start_ref = oos_start

        n_months = len(raw_w)
        n_tickers = (raw_w != 0).any().sum()
        print(
            f"\n  [{label}]  {oos_start.date()} – {raw_w.index[-1].date()}"
            f"  |  {n_months} signal months  |  {n_tickers} active tickers"
        )
        res = _run_engine(raw_w, pnl_prices, tcost)
        res["name"] = label
        all_results[label] = res

    # SPY B&H — also through BACKTEST.py: one rebalance at the start, then hold.
    spy_monthly_px = spy_prices.resample("ME").last()
    spy_bt         = spy_monthly_px.loc[oos_start_ref:]
    spy_w          = pd.DataFrame(1.0, index=[spy_bt.index[0]], columns=["SPY"])
    spy_res        = BACKTEST.backtest(
        spy_w, spy_bt,
        freq=12, lag=0,
        transaction_cost=0.0,
        signal_dates=[spy_bt.index[0]],
    )
    spy_res["name"] = "S&P 500 B&H"
    all_results["S&P 500 B&H"] = spy_res

    # ── 6. Comparison table + equity chart ───────────────────────────────────
    print("\n" + "═" * 70)
    years = round((data_end - oos_start_ref).days / 365.25, 1)
    analysis = BACKTEST.results_backtest(
        all_results,
        title=f"Deep Momentum vs Bench vs S&P 500 B&H  ({years}y, {oos_start_ref.year}–present)",
    )

    fig = analysis["fig"]
    fig.savefig(save_fig, dpi=150, bbox_inches="tight")
    print(f"\n[plot] saved → {save_fig}")

    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(analysis["summary_df"].to_string())

    if len(analysis["yearly_df"]) > 0:
        print("\n── Yearly Returns ───────────────────────────────────────────────────")
        ret_cols = [c for c in analysis["yearly_df"].columns if "Return" in c]
        print(analysis["yearly_df"][ret_cols].applymap(lambda x: f"{x:.1%}" if pd.notna(x) else "").to_string())

    return all_results


if __name__ == "__main__":
    import os
    args = sys.argv[1:]

    if args and args[0] == "--compare-tiingo":
        # python deep_momentum_xgb.py --compare-tiingo [start_date] [n_seeds]
        start_yr    = args[1] if len(args) > 1 else "2000-01-01"
        seeds       = int(args[2]) if len(args) > 2 else N_SEEDS
        ckpt_exists = os.path.exists("tiingo_download_checkpoint.parquet")
        print(f"\n[tiingo] Loading broad universe (start={start_yr}, seeds={seeds}) …")
        data = load_broad_universe_tiingo(start_date=start_yr, skip_download=ckpt_exists)
        compare_strategies(data, n_seeds=seeds)

    elif args and os.path.isfile(args[0]) and args[0].endswith(".csv"):
        # CSV archive pipeline: python deep_momentum_xgb.py all_stocks_5yr.csv srp ls
        csv   = args[0]
        strat = args[1] if len(args) > 1 else "srp"
        port  = args[2] if len(args) > 2 else "ls"
        run_with_backtest(csv, strategy=strat, portfolio=port, min_train_months=18, n_seeds=3)

    else:
        backtest(args[0] if args else None)

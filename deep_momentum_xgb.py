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

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MOM_WINDOWS   = [1, 3, 6, 9, 12]   # momentum lookback months
N_SEEDS       = 20                  # ensemble size (paper uses 50-100; 20 balances speed vs. stability)
MIN_TRAIN_YRS = 10                  # minimum training history before first pred
TOP_Q         = 0.05                # long / short tail size (paper: edge concentrates in outer 5%)

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


# ── 2. Feature engineering ────────────────────────────────────────────────────
def make_features(
    rets: pd.DataFrame,
    size: pd.DataFrame,
    t: int,
    ffd_scores: dict | None = None,
) -> pd.DataFrame:
    """
    Build feature matrix at time index t.  Paper Table 1 + dynamics (+ FFD).

    Paper features (20): 5 nMOM + 5 MMOM + 10 SIZE dummies (MOM-SZ-NOM spec).
    Added momentum-dynamics features (6), each z-scored cross-sectionally with its
    cross-sectional mean retained (same treatment as nMOM/MMOM, paper Sec. 3.2.3):
        ACCEL  recent (t-6..t-2) minus older (t-12..t-7) cumulative return  → rising/fading momentum
        VOL    trailing 11-month cross-sectional realized vol               → momentum-risk conditioning
        POS    fraction of up months over t-12..t-2                         → frog-in-the-pan consistency
    If ffd_scores is supplied, 10 more (5 zFFD + 5 MFFD) are appended → 36 total.
    All computed with one vectorized op over the cross-section (no Python loops over stocks).
    """
    feats = {}

    def _zc(s: pd.Series, name: str):
        """Cross-sectional z-score + retain the cross-sectional mean (macro state)."""
        mu, sigma = s.mean(), s.std()
        feats[f"z{name}"] = (s - mu) / (sigma + 1e-10)
        feats[f"M{name}"] = mu

    # ── Raw cumulative-return momentum (paper Eq. 6–8) — ALWAYS ───────────────
    for m in MOM_WINDOWS:
        if m == 1:
            mom = rets.iloc[t - 1]
        else:
            mom = (1 + rets.iloc[t - m : t - 1]).prod() - 1
        mu    = mom.mean()
        sigma = mom.std()
        feats[f"zMOM{m}"] = (mom - mu) / (sigma + 1e-10)
        feats[f"MMOM{m}"] = mu

    # ── Momentum-dynamics features (all vectorized across stocks) — ALWAYS ────
    win = rets.iloc[t - 12 : t - 1]                          # months t-12..t-2 (11 rows)
    mom_recent = (1 + rets.iloc[t - 6  : t - 1]).prod() - 1  # t-6..t-2
    mom_older  = (1 + rets.iloc[t - 12 : t - 6]).prod() - 1  # t-12..t-7
    _zc(mom_recent - mom_older, "ACCEL")                     # momentum acceleration
    _zc(win.std(),              "VOL")                       # trailing realized vol
    _zc((win > 0).mean(),       "POS")                       # fraction of up months

    # ── FFD momentum (López de Prado, AFML Ch. 5) — ADDED when scores supplied ─
    # Stationary, memory-preserving trend.  No look-ahead: optimal d is frozen on
    # the pre-OOS training window and the FIR filter is causal (see
    # _ffd_from_training_window).  Each window z-scored cross-sectionally + mean kept.
    if ffd_scores is not None:
        for m in sorted(ffd_scores.keys()):     # decoupled from MOM_WINDOWS (e.g. {1,3,12})
            row   = ffd_scores[m].iloc[t - 1].reindex(rets.columns)
            mu    = row.mean()
            sigma = row.std()
            feats[f"zFFD{m}"] = (row - mu) / (sigma + 1e-10)
            feats[f"MFFD{m}"] = mu

    # SIZE: 10 binary dummies D_s, s=1..10 (paper Sec. 3.3.1)
    cap         = size.iloc[t - 1].rank(pct=True, na_option="keep").fillna(0.5)
    size_decile = ((cap * 9.999).astype(int) + 1)  # 1–10
    for s in range(1, 11):
        feats[f"SIZE_{s}"] = (size_decile == s).astype(float)

    return pd.DataFrame(feats, index=rets.columns)


# ── 3. Label construction ─────────────────────────────────────────────────────
def make_labels(rets: pd.DataFrame, t: int) -> pd.Series:
    """
    Next-month cross-sectional return decile, 0-indexed.
    Label 0 = highest return decile, label 9 = lowest.

    Paper formula: label = 10 - qcut(..., labels=False)  → 0-indexed via -1.
    Equivalent: 9 - qcut(..., labels=False).
    """
    fwd = rets.iloc[t].dropna()
    return (9 - pd.qcut(fwd, 10, labels=False, duplicates="drop")).astype(int)


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
        L   = make_labels(rets, t)
        idx = (
            F.index
             .intersection(L.index)
             .intersection(rets.iloc[t].dropna().index)
        )
        if len(idx) >= 20:
            pool[t] = (F.loc[idx], L.loc[idx])

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

    # Coverage filter: fraction of the stock's own active months with price > min_price.
    # A stock active 2005-2010 only needs 70% of THOSE months, not 70% of all 318.
    # This preserves delisted stocks for survivorship-bias-free analysis.
    has_price  = close_wide > min_price
    n_active   = has_price.notna().sum()          # months where row exists
    n_good_px  = (has_price == True).sum()        # months with price > min_price
    # Require at least 12 months of data AND min_coverage within active window
    good       = (n_active >= 12) & (n_good_px / n_active.clip(lower=1) >= min_coverage)
    close_wide  = close_wide.loc[:, good]
    volume_wide = volume_wide.reindex(columns=close_wide.columns)

    if verbose:
        print(f"[tiingo] kept {good.sum():,}/{len(good):,} tickers "
              f"(≥12 months data, price≥${min_price} in ≥{min_coverage:.0%} of active months)")

    prices_monthly = close_wide
    volume_monthly = volume_wide
    # Return cleaning (paper §4.1): drop data errors (>+300% / <-95%), then winsorize
    # at the CROSS-SECTIONAL 1/99 percentile each month.  Unlike a fixed ±50% clip,
    # this stays wide during momentum-crash months (when many losers rebound together),
    # so the short-side crash risk that should punish raw momentum is preserved.
    _raw           = prices_monthly.pct_change()
    _raw           = _raw.mask((_raw > 3.0) | (_raw < -0.95))
    _lo            = _raw.quantile(0.01, axis=1)
    _hi            = _raw.quantile(0.99, axis=1)
    rets_monthly   = _raw.clip(_lo, _hi, axis=0)
    size_monthly   = (prices_monthly * volume_monthly).ffill()

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
        L   = make_labels(rets, t)
        idx = (
            F.index
             .intersection(L.index)
             .intersection(rets.iloc[t].dropna().index)
        )
        if len(idx) >= 20:
            pool[t] = (F.loc[idx], L.loc[idx])

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
        compute_risk_metrics=False,   # skip expensive MRC loop for PoC speed
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
) -> pd.DataFrame:
    """
    Cross-sectional momentum long(/short) — no XGBoost.
    ffd_scores: pre-computed FFD scores (skips re-estimation if provided).
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
        if len(F) < 20:
            continue
        signal_date = rets.index[t - 1]
        s_ser = F[signal_col]
        n     = max(1, int(len(s_ser) * q))
        w     = pd.Series(0.0, index=F.index)
        w[s_ser.nlargest(n).index]  = +1.0 / n
        if portfolio == "ls":
            w[s_ser.nsmallest(n).index] = -1.0 / n
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
    max_train_months: int = 60,
    n_seeds: int = N_SEEDS,
    use_ffd: bool = True,
    ffd_scores: dict | None = None,
    pool: dict | None = None,
) -> dict:
    """
    Train XGBoost (rolling window), then score DPR / RET / SRP.
    max_train_months: cap on training window size (rolling, not expanding).
    ffd_scores / pool: pass pre-computed values to skip recomputation.
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
            L   = make_labels(rets, t)
            idx = (
                F.index
                 .intersection(L.index)
                 .intersection(rets.iloc[t].dropna().index)
            )
            if len(idx) >= 20:
                pool[t] = (F.loc[idx], L.loc[idx])

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
            F = make_features(rets, size, t, ffd_scores=ffd_scores).dropna()
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
                w[s_ser.nsmallest(n).index] = -1.0 / n
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


def _monthly_ls_backtest(
    weights: pd.DataFrame,
    rets: pd.DataFrame,
    *,
    transaction_cost: float = 0.001,
    freq: int = 12,
) -> dict:
    """
    Direct monthly L/S portfolio backtest — no daily drift, no leverage blowup.

    weights : DataFrame indexed by signal dates (month-end of t-1).
              Each row should have long weights summing to +1 and short to -1.
    rets    : Monthly returns DataFrame, indexed by month-end of t.

    For each signal date the portfolio is held during the NEXT calendar month,
    using that month's realised returns.  Transaction cost is applied as a
    fraction of one-way notional traded (weight changes vs. previous period).
    """
    port_rets: list[float] = []
    dates:     list[pd.Timestamp] = []
    turnovers: list[float] = []
    w_prev: pd.Series | None = None

    for sig_date in weights.index:
        pos = rets.index.searchsorted(sig_date, side="right")
        if pos >= len(rets):
            continue
        hold_date = rets.index[pos]

        w = weights.loc[sig_date]
        r = rets.loc[hold_date].reindex(w.index).fillna(0.0)

        port_r = float((w * r).sum())

        # one-way turnover relative to previous portfolio
        if w_prev is not None:
            delta = (w - w_prev.reindex(w.index, fill_value=0.0)).abs().sum()
            one_way = 0.5 * float(delta)
            turnovers.append(one_way)
            port_r -= transaction_cost * one_way

        w_prev = w.copy()
        dates.append(hold_date)
        port_rets.append(port_r)

    if not dates:
        raise ValueError("_monthly_ls_backtest: no valid holding months found.")

    returns  = pd.Series(port_rets, index=pd.DatetimeIndex(dates), name="returns")
    equity   = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    n          = len(returns)
    ann_return = float(equity.iloc[-1] ** (freq / n) - 1.0)
    ann_vol    = float(returns.std() * np.sqrt(freq))
    sharpe     = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd     = float(drawdown.min())
    avg_dd     = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0
    ann_to     = float(np.mean(turnovers) * freq) if turnovers else np.nan
    total_ret  = float(equity.iloc[-1] - 1.0)

    return {
        "returns":           returns,
        "equity":            equity,
        "drawdown":          drawdown,
        "ann_return":        ann_return,
        "ann_vol":           ann_vol,
        "sharpe":            sharpe,
        "max_drawdown":      max_dd,
        "avg_drawdown":      avg_dd,
        "total_return":      total_ret,
        "ann_turnover":      ann_to,
        # placeholder keys expected by results_backtest summary table
        "cdar":              np.nan,
        "cvar_ann":          np.nan,
        "downside_deviation": np.nan,
        "mrc_variance":      np.nan,
        "herfindahl":        np.nan,
        "expectancy":        np.nan,
        "total_cost":        np.nan,
    }


def compare_strategies(
    preloaded: tuple,
    *,
    n_seeds: int = N_SEEDS,
    min_train_months: int = MIN_TRAIN_YRS * 12,
    max_train_months: int = 60,
    q: float = TOP_Q,
    transaction_cost: float = 0.001,
    save_fig: str = "dm_comparison_tiingo.png",
) -> dict:
    """
    Run Bench (zMOM12 L/S), DM-DPR, DM-RET, DM-SRP L/S plus SPY B&H.
    preloaded : (prices_monthly, rets_monthly, size_monthly) from load_broad_universe_tiingo()
    Returns dict  label → backtest result dict.
    """
    import matplotlib
    matplotlib.use("Agg")

    try:
        import BACKTEST
        import yfinance as yf
    except ImportError as e:
        raise ImportError(str(e))

    # ── 1. Unpack preloaded Tiingo data ───────────────────────────────────────
    prices_monthly, rets_monthly, size_monthly = preloaded
    data_start = rets_monthly.index[0]
    data_end   = rets_monthly.index[-1]

    # ── 2. Bench weights (raw zMOM12 L/S) ────────────────────────────────────
    print("\n── Bench zMOM12 L/S ─────────────────────────────────────────────────")
    bench_w = _bench_weights(
        rets_monthly, size_monthly, prices_monthly,
        min_train_months=min_train_months, q=q, use_ffd=False,
        portfolio="ls",
    )

    # ── 3. DM weights (momentum + dynamics + FFD features) ───────────────────
    # use_ffd=True: optimal d frozen on the initial training window, causal filter
    # → FFD features added with no OOS look-ahead.
    print("\n── Deep Momentum L/S (+ FFD) ────────────────────────────────────────")
    dm_w = _generate_all_dm_weights(
        rets_monthly, size_monthly, prices_monthly,
        min_train_months=min_train_months, max_train_months=max_train_months,
        q=q, n_seeds=n_seeds, use_ffd=True,
        portfolio="ls", pool=None,
    )

    # ── 4. SPY download ──────────────────────────────────────────────────────
    spy_raw = yf.download(
        "SPY",
        start=data_start.strftime("%Y-%m-%d"),
        end=(data_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False,
    )
    # yfinance may return MultiIndex (metric, ticker) or flat columns depending on version
    spy_close = spy_raw["Close"]
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_prices = spy_close.to_frame("SPY")
    spy_prices.index = pd.to_datetime(spy_prices.index)

    # ── 5. Backtest each strategy (monthly, no daily drift) ──────────────────
    # Use _monthly_ls_backtest to avoid gross-leverage blowup from daily drift
    # between monthly rebalances (the drift mode allows leverage > 2× to accumulate).
    all_results = {}
    oos_start_ref = None

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
        res = _monthly_ls_backtest(
            raw_w, rets_monthly,
            transaction_cost=transaction_cost,
            freq=12,
        )
        res["name"] = label
        all_results[label] = res

    # SPY B&H — monthly, aligned to first HOLDING month (one past signal date)
    spy_monthly = spy_prices.resample("ME").last().pct_change().dropna()
    spy_monthly.columns = ["SPY"]
    spy_oos_pos = spy_monthly.index.searchsorted(oos_start_ref, side="right")
    spy_ret_oos = spy_monthly.iloc[spy_oos_pos:]["SPY"]
    spy_equity  = (1 + spy_ret_oos).cumprod()
    spy_dd      = spy_equity / spy_equity.cummax() - 1
    n_spy       = len(spy_ret_oos)
    spy_ann_ret = float(spy_equity.iloc[-1] ** (12 / n_spy) - 1.0) if n_spy > 0 else np.nan
    spy_ann_vol = float(spy_ret_oos.std() * np.sqrt(12))
    spy_sharpe  = spy_ann_ret / spy_ann_vol if spy_ann_vol > 0 else np.nan
    spy_res = {
        "name":              "S&P 500 B&H",
        "returns":           spy_ret_oos,
        "equity":            spy_equity,
        "drawdown":          spy_dd,
        "ann_return":        spy_ann_ret,
        "ann_vol":           spy_ann_vol,
        "sharpe":            spy_sharpe,
        "max_drawdown":      float(spy_dd.min()),
        "avg_drawdown":      float(spy_dd[spy_dd < 0].mean()) if (spy_dd < 0).any() else 0.0,
        "total_return":      float(spy_equity.iloc[-1] - 1.0),
        "ann_turnover":      0.0,
        "cdar":              np.nan,
        "cvar_ann":          np.nan,
        "downside_deviation": np.nan,
        "mrc_variance":      np.nan,
        "herfindahl":        np.nan,
        "expectancy":        np.nan,
        "total_cost":        0.0,
    }
    all_results["S&P 500 B&H"] = spy_res

    # ── 6. Comparison table + equity chart ───────────────────────────────────
    print("\n" + "═" * 70)
    years = round((data_end - oos_start_ref).days / 365.25, 1)
    analysis = BACKTEST.results_backtest(
        all_results,
        title=f"Deep Momentum vs Bench vs S&P 500 B&H  ({years}y, {oos_start_ref.year}–present)",
        fama_french=False,
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
        compare_strategies(data, n_seeds=seeds, max_train_months=60)

    elif args and os.path.isfile(args[0]) and args[0].endswith(".csv"):
        # CSV archive pipeline: python deep_momentum_xgb.py all_stocks_5yr.csv srp ls
        csv   = args[0]
        strat = args[1] if len(args) > 1 else "srp"
        port  = args[2] if len(args) > 2 else "ls"
        run_with_backtest(csv, strategy=strat, portfolio=port, min_train_months=18, n_seeds=3)

    else:
        backtest(args[0] if args else None)

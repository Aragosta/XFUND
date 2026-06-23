#!/usr/bin/env python3
"""
Deep Momentum XGBoost — Han (2022) proof of concept.

Two-step approach:
  1. XGBoost multiclass classifier predicts next-month return decile probabilities.
  2. Three reclassification scores (DPR / RET / SRP) re-rank stocks to filter
     out the bimodality-driven crash tail of momentum strategies.

Usage:
    python deep_momentum_xgb.py                   # synthetic data
    python deep_momentum_xgb.py sp500_stocks.csv  # Kaggle S&P 500 format
                                                  # (columns: date,open,high,
                                                  #  low,close,volume,Name)
"""
import sys
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MOM_WINDOWS   = [1, 3, 6, 9, 12]   # momentum lookback months
N_SEEDS       = 10                  # ensemble size
MIN_TRAIN_YRS = 10                  # minimum training history before first pred
VAL_YRS       = 10                  # trailing validation window for early stopping
TOP_Q         = 0.10                # long / short decile size

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    num_class=10,
    objective="multi:softprob",
    eval_metric="mlogloss",
    early_stopping_rounds=20,
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
def make_features(rets: pd.DataFrame, size: pd.DataFrame, t: int) -> pd.DataFrame:
    """
    Build (N_stocks × 20) feature matrix at time index t.

    Momentum (JT 1-month skip convention):
        MOM_1  = r_{t-1}
        MOM_m  = cumulative product  r_{t-m} … r_{t-2}   (skip r_{t-1})

    Cross-sectional normalisation per month:
        zMOM_m = (MOM_m - mean) / std     — idiosyncratic signal
        MMOM_m = mean                     — macro / cross-sectional level

    Size: one-hot decile from market-cap proxy  →  size_d1 … size_d10
    """
    feats = {}
    for m in MOM_WINDOWS:
        if m == 1:
            mom = rets.iloc[t - 1]
        else:
            # cumulative product of (m-1) months, skipping r_{t-1}
            mom = (1 + rets.iloc[t - m : t - 1]).prod() - 1
        mu              = mom.mean()
        feats[f"zMOM{m}"] = (mom - mu) / (mom.std() + 1e-10)
        feats[f"MMOM{m}"] = mu   # scalar → broadcast to all stocks

    cap    = size.iloc[t - 1].rank(pct=True, na_option="keep").fillna(0.5)
    decile = ((cap * 10).clip(upper=9.999)).astype(int)
    for d in range(10):
        feats[f"size_d{d + 1}"] = (decile == d).astype(float)

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

        # ── Annual retraining with expanding window ────────────────────────
        # Validation: trailing VAL_YRS of training data (chronological split)
        val_t_start = t_cut - VAL_YRS * 12

        tr_ts = [t for t in pool if first_feat <= t < val_t_start]
        va_ts = [t for t in pool if val_t_start <= t < t_cut]

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
    # Adaptive validation split: last third of training window (minimum 6 months)
    val_months = max(6, min_train_months // 3)

    if first_pred >= T - 1:
        raise ValueError(
            f"Need ≥ {first_pred + 1} monthly observations; got {T}. "
            f"Reduce min_train_months (currently {min_train_months})."
        )

    # Pre-compute features and labels once
    pool = {}
    for t in range(first_feat, T - 1):
        F   = make_features(rets, size, t).dropna()
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
        t_cut       = months[0]
        val_t_start = t_cut - val_months

        tr_ts = [t for t in pool if first_feat <= t < val_t_start]
        va_ts = [t for t in pool if val_t_start <= t < t_cut]

        if len(tr_ts) >= 12 and len(va_ts) >= 6:
            X_tr = pd.concat([pool[t][0] for t in tr_ts])
            y_tr = pd.concat([pool[t][1] for t in tr_ts])
            X_va = pd.concat([pool[t][0] for t in va_ts])
            y_va = pd.concat([pool[t][1] for t in va_ts])

            fwd_tr  = pd.concat([rets.iloc[t].reindex(pool[t][0].index) for t in tr_ts])
            lab_tr  = pd.concat([pool[t][1] for t in tr_ts])
            grp     = pd.DataFrame({"l": lab_tr.values, "r": fwd_tr.values}).groupby("l")["r"]
            mu_k    = np.array([grp.get_group(k).mean() if k in grp.groups else 0.0  for k in range(10)])
            sigma2_k = np.array([grp.get_group(k).var()  if k in grp.groups else 1e-4 for k in range(10)])

            print(f"  [{year}] train={len(X_tr):,}  val={len(X_va):,} obs  — fitting {n_seeds} models")
            xgb_p = {**XGB_PARAMS, "n_estimators": min(XGB_PARAMS["n_estimators"], 200)}
            models = [
                XGBClassifier(**xgb_p, random_state=s).fit(
                    X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False
                )
                for s in range(n_seeds)
            ]
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


if __name__ == "__main__":
    import os
    args = sys.argv[1:]
    if args and os.path.isfile(args[0]) and args[0].endswith(".csv"):
        # Run the BACKTEST.py-integrated pipeline on the archive
        # Example: python deep_momentum_xgb.py all_stocks_5yr.csv srp ls
        csv   = args[0]
        strat = args[1] if len(args) > 1 else "srp"
        port  = args[2] if len(args) > 2 else "ls"
        run_with_backtest(csv, strategy=strat, portfolio=port, min_train_months=18, n_seeds=3)
    else:
        backtest(args[0] if args else None)

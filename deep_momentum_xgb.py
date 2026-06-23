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


if __name__ == "__main__":
    backtest(sys.argv[1] if len(sys.argv) > 1 else None)

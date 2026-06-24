#!/usr/bin/env python3
"""
Load Tiingo checkpoint and run full Deep Momentum comparison.
Usage: python run_tiingo_analysis.py
"""
import pandas as pd
from deep_momentum_xgb import compare_strategies

CHECKPOINT = "tiingo_download_checkpoint.parquet"
MIN_PRICE    = 1.0
MIN_COVERAGE = 0.70

print(f"[load] reading {CHECKPOINT} …")
ckpt = pd.read_parquet(CHECKPOINT)
close_wide  = ckpt["close"]
volume_wide = ckpt["volume"]

has_price   = close_wide > MIN_PRICE
n_active    = has_price.notna().sum()
n_good_px   = (has_price == True).sum()
good        = (n_active >= 12) & (n_good_px / n_active.clip(lower=1) >= MIN_COVERAGE)
close_wide  = close_wide.loc[:, good]
volume_wide = volume_wide.reindex(columns=close_wide.columns)

prices_monthly = close_wide.sort_index()
volume_monthly = volume_wide.sort_index()
rets_monthly   = prices_monthly.pct_change(fill_method=None).clip(-0.5, 0.5)
size_monthly   = (prices_monthly * volume_monthly).ffill()

print(f"[load] {prices_monthly.shape[1]:,} tickers  |  "
      f"{prices_monthly.index[0].date()} – {prices_monthly.index[-1].date()}  |  "
      f"{len(prices_monthly)} months")

compare_strategies(
    (prices_monthly, rets_monthly, size_monthly),
    n_seeds=20,
    min_train_months=120,   # 10-year warmup per Han (2022)
    save_fig="dm_comparison_tiingo.png",
)

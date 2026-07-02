#!/usr/bin/env python3
"""Reconstruct a monthly panel (adjClose + dollar-volume) from the full daily download.
Enables DM (needs monthly close + $-volume/size) and LM (needs daily) on the SAME broad universe.

  reads : tiingo_daily_close.parquet, tiingo_daily_volume.parquet   (partial OK)
  writes: tiingo_monthly_from_daily.parquet  (MultiIndex cols: 'close' | 'dollarvol')
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd

c = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
v = pd.read_parquet("tiingo_daily_volume.parquet").sort_index().reindex_like(c)
print(f"[recon] daily close {c.shape}  {c.index.min().date()}->{c.index.max().date()}", flush=True)

me = c.resample("ME")
close_m  = me.last()                                    # month-end adjClose
dvol_m   = (c * v).resample("ME").sum()                 # total $ traded in month (dollar volume)
out = pd.concat({"close": close_m, "dollarvol": dvol_m}, axis=1)
out.to_parquet("tiingo_monthly_from_daily.parquet")
print(f"[recon] monthly {close_m.shape}  {close_m.index.min().date()}->{close_m.index.max().date()}", flush=True)
print(f"[recon] avg names/month with $vol>1e6 (2011+): "
      f"{(dvol_m.loc['2011':] > 1e6).sum(axis=1).mean():.0f}", flush=True)
print("[recon] -> tiingo_monthly_from_daily.parquet", flush=True)

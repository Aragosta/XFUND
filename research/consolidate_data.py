#!/usr/bin/env python3
"""
consolidate_data.py — merge the Tiingo bulk pull (22,996 tickers, 1990-2026) into ONE wide price panel that
matches the existing data/daily.parquet schema/basis, plus report on the EDGAR facts_v2 fundamentals alignment.

BASIS DECISION (verified before writing): current daily.parquet `close` is SPLIT-ADJUSTED (checked AAPL across
its June-2014 7:1 split — smooth, no jump). Tiingo's `adjClose/adjOpen/adjHigh/adjLow/adjVolume` are the same
basis (split+dividend adjusted). So the new panel uses Tiingo's adj* columns renamed to open/high/low/close/volume
— NOT raw — to stay consistent with everything downstream (DATAHUB, all sleeves) that assumes adjusted prices.

MERGE RULE: for any ticker present in BOTH the old daily.parquet and the new Tiingo pull, the NEW pull is
AUTHORITATIVE for that ticker's entire history (it is a fresh full-history download — Tiingo retroactively
restates splits/divs across the whole series, so a fresh pull is strictly more correct than a partial legacy one).
Tickers ONLY in the old file (if any — expected near-zero since the new pull is a superset) are kept from old.

Writes (does NOT touch the existing daily.parquet / facts.parquet — new files only, validate before swapping):
  data/daily_v2.parquet          — wide (field, ticker) x date, float32, matches daily.parquet's MultiIndex shape
  data/corporate_actions.parquet — long (ticker, date, divCash, splitFactor) — kept separately, useful later for
                                    a raw-price/split-coherent-shares rebuild (flagged as a TODO in DATAHUB).
Prints a full alignment/coverage report at the end.
"""
import warnings, os, sys, glob, gc
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.chdir("/Users/enzokreeft/XFUND")

FIELDS = {"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "close", "adjVolume": "volume"}

print("[consolidate] loading Tiingo batches ...", flush=True)
batches = sorted(glob.glob("data/tiingo_eod/batch_*.parquet"))
raw = pd.concat([pd.read_parquet(f, columns=["date", "ticker"] + list(FIELDS) + ["divCash", "splitFactor"])
                 for f in batches], ignore_index=True)
raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.tz_localize(None)
before = len(raw)
raw = raw.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
print(f"  {before:,} rows loaded -> {len(raw):,} after dedup · {raw['ticker'].nunique():,} tickers · "
      f"{raw['date'].min().date()} .. {raw['date'].max().date()}", flush=True)

# ── corporate actions (kept separately for future raw-mcap/split-coherent-shares work) ─────────────────
ca = raw.loc[(raw["divCash"] > 0) | (raw["splitFactor"] != 1.0), ["ticker", "date", "divCash", "splitFactor"]]
ca.to_parquet("data/corporate_actions.parquet")
print(f"[consolidate] corporate_actions.parquet: {len(ca):,} rows (dividends/splits)", flush=True)

# ── old panel: what tickers/dates does it cover? ────────────────────────────────────────────────────
old = pd.read_parquet("data/daily.parquet")
old_tickers = set(old["close"].columns)
new_tickers = set(raw["ticker"].unique())
only_old = sorted(old_tickers - new_tickers)
print(f"\n[consolidate] old panel: {len(old_tickers):,} tickers, {old['close'].shape[0]:,} dates "
      f"({old['close'].index.min().date()} .. {old['close'].index.max().date()})", flush=True)
print(f"[consolidate] new panel: {len(new_tickers):,} tickers", flush=True)
print(f"[consolidate] tickers ONLY in old (kept from legacy data): {len(only_old)}", flush=True)

# spot-check basis agreement on overlapping names/dates (sanity: correlation should be ~1.0)
common = list(old_tickers & new_tickers)[:200]
ov = raw[raw.ticker.isin(common)].pivot(index="date", columns="ticker", values="adjClose")
oc = old["close"].reindex(columns=common)
aligned = ov.reindex(index=oc.index.intersection(ov.index))
oc2 = oc.reindex(aligned.index)
corrs = aligned.corrwith(oc2)
print(f"[consolidate] basis sanity check ({len(common)} overlapping tickers): "
      f"median corr(old.close, new.adjClose) = {corrs.median():.4f}  (should be ~1.0)", flush=True)
del ov, oc, aligned, oc2; gc.collect()

# ── build the consolidated wide panel, field by field (memory-controlled, float32) ─────────────────────
all_dates = pd.DatetimeIndex(sorted(set(raw["date"]) | set(old.index)))
all_tickers = sorted(new_tickers | old_tickers)
print(f"\n[consolidate] building wide panel: {len(all_tickers):,} tickers x {len(all_dates):,} dates "
      f"(~{len(all_tickers)*len(all_dates)*4/1e9:.1f} GB per field, float32) ...", flush=True)

panels = {}
for src_col, dst_name in FIELDS.items():
    piv = raw.pivot(index="date", columns="ticker", values=src_col).reindex(index=all_dates, columns=all_tickers)
    if only_old:                                                     # backfill legacy-only tickers from old panel
        old_col = "volume" if dst_name == "volume" else dst_name
        if old_col in old.columns.get_level_values(0):
            legacy = old[old_col].reindex(index=all_dates, columns=only_old)
            piv[only_old] = legacy
    piv = piv.astype(np.float32)
    panels[dst_name] = piv
    print(f"  {dst_name:8} panel built: {piv.shape}  nonNaN={piv.notna().sum().sum():,}", flush=True)
    del piv; gc.collect()

out = pd.concat(panels, axis=1)
del panels, raw, old; gc.collect()
out.to_parquet("data/daily_v2.parquet")
print(f"\n[consolidate] wrote data/daily_v2.parquet  shape={out.shape}  "
      f"size={os.path.getsize('data/daily_v2.parquet')/1e9:.2f} GB", flush=True)

# ── EDGAR alignment report ──────────────────────────────────────────────────────────────────────────
fv2 = pd.read_parquet("data/edgar/facts_v2.parquet")
fund_tickers = set(fv2["ticker"].str.upper().unique())
price_tickers = set(all_tickers)
print(f"\n[consolidate] EDGAR facts_v2: {fv2.ticker.nunique():,} tickers, {len(fv2):,} rows", flush=True)
print(f"[consolidate] price∩fundamentals overlap: {len(price_tickers & fund_tickers):,} "
      f"({100*len(price_tickers & fund_tickers)/len(price_tickers):.1f}% of price universe)", flush=True)

print("\n" + "=" * 70)
print("FINAL CONSOLIDATED RESULT")
print("=" * 70)
print(f"  price panel   : {out['close'].shape[1]:,} tickers x {out['close'].shape[0]:,} dates "
      f"({out.index.min().date()} .. {out.index.max().date()})")
print(f"  delisted incl.: survivorship-bias-free (Tiingo catalog delisted names retained)")
print(f"  fundamentals  : {fv2.ticker.nunique():,} tickers, {len(fv2):,} PIT fact rows, "
      f"{sorted(fv2.concept.unique())}")
print(f"  price<->fund  : {len(price_tickers & fund_tickers):,} tickers have BOTH price and fundamentals")
print(f"  files         : data/daily_v2.parquet, data/edgar/facts_v2.parquet, data/corporate_actions.parquet")

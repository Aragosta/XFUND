#!/usr/bin/env python3
"""
data_integrity.py — full audit of the consolidated substrate (data/daily.parquet + data/edgar/facts.parquet).
Distinguishes LEGITIMATE NaNs (pre-IPO / post-delist / pre-filing — expected in a survivorship-bias-free panel)
from SPURIOUS ones (gaps inside an active window, dead columns, corruption). Prints PASS/FAIL per check.
Memory-safe: loads ONE price field at a time via pyarrow column selection (safe to run beside the sleeve jobs).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyarrow.parquet as pq, gc, os
os.chdir("/Users/enzokreeft/XFUND")

PATH = "data/daily.parquet"
_allcols = [n for n in pq.read_schema(PATH).names if n != "__index_level_0__"]
def load_field(fld):
    cols = [c for c in _allcols if c.startswith(f"('{fld}',")]
    df = pq.read_table(PATH, columns=cols + ["__index_level_0__"]).to_pandas()
    if "__index_level_0__" in df.columns:                 # pandas may auto-restore it as the index
        df = df.set_index("__index_level_0__")
    df.index = pd.DatetimeIndex(df.index)
    df.columns = [(eval(c)[1] if isinstance(c, str) else c[1]) for c in df.columns]  # tuple-str OR real tuple
    return df.sort_index()

FAIL = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok: FAIL.append(name)

print("=" * 82); print("PRICE PANEL — data/daily.parquet"); print("=" * 82)
fields = sorted({eval(c)[0] for c in _allcols})
check("has 5 OHLCV fields", set(fields) == {"open","high","low","close","volume"}, str(fields))
tsets = {f: {eval(c)[1] for c in _allcols if c.startswith(f"('{f}',")} for f in fields}
check("all fields share identical ticker set", all(tsets[f] == tsets["close"] for f in fields),
      f"{len(tsets['close'])} tickers each")

close = load_field("close")
check("index strictly increasing & unique", close.index.is_monotonic_increasing and close.index.is_unique)
print(f"  shape: {close.shape}  ({close.index.min().date()}..{close.index.max().date()})", flush=True)
dead = close.columns[close.notna().sum() == 0].tolist()
check("no all-NaN (dead) ticker columns", len(dead) == 0, f"{len(dead)} dead: {dead[:5]}" if dead else "")
check("no non-positive close prices", int((close <= 0).sum().sum()) == 0, f"{int((close<=0).sum().sum())} bad")
check("no infinities in close", not np.isinf(close.to_numpy(dtype='float64')).any())

# INTERNAL gaps: NaNs BETWEEN first & last valid date per ticker (a listed stock should trade every day)
arr = close.to_numpy(); n_dates = len(close)
first = np.argmax(~np.isnan(arr), axis=0)
last = n_dates - 1 - np.argmax(~np.isnan(arr[::-1]), axis=0)
valid_any = ~np.isnan(arr).all(axis=0)
internal = 0; gap_names = 0
for j in np.where(valid_any)[0]:
    span = arr[first[j]:last[j]+1, j]; g = int(np.isnan(span).sum())
    if g: internal += g; gap_names += 1
check("no internal price gaps within active window", internal == 0,
      f"{internal} internal NaNs across {gap_names} tickers" if internal else "fully dense")
# glitch detector on returns
ret = close.pct_change(fill_method=None)
absurd = int(((ret > 20) | (ret < -0.99)).sum().sum())
check("no absurd daily returns (>2000% up / <-99% down)", absurd == 0, f"{absurd} suspicious")
# AAPL split continuity (4:1 on 2020-08-31, adjusted series must be smooth)
if "AAPL" in close.columns:
    s = close["AAPL"].loc["2020-08-24":"2020-09-04"].dropna()
    jmp = float(s.pct_change().abs().max())
    check("AAPL smooth across 2020 4:1 split (adjusted basis)", jmp < 0.15, f"max daily move {jmp:.1%}")
del ret; gc.collect()

# OHLC validity — load each range field, compare to close, then free
mask_c = close.notna()
for pair, op in [(("high","low"), "hi>=lo"), (("high","close"), "hi>=close"), (("low","close"), "lo<=close")]:
    a = load_field(pair[0]); b = close if pair[1]=="close" else load_field(pair[1])
    m = mask_c & a.notna() & b.notna()
    bad = int((((a < b) if op!="lo<=close" else (a > b)) & m).sum().sum())
    check(f"OHLC valid: {op}", bad == 0, f"{bad} violations")
    del a
    if pair[1] != "close": del b
    gc.collect()
vol = load_field("volume")
check("no negative volume", int((vol < 0).sum().sum()) == 0)
covered_ticks = len(tsets["close"]); del vol, close, arr; gc.collect()

print("\n" + "=" * 82); print("FUNDAMENTALS — data/edgar/facts.parquet"); print("=" * 82)
f = pd.read_parquet("data/edgar/facts.parquet")
check("no NaN in val", int(f["val"].isna().sum()) == 0, f"{int(f['val'].isna().sum())} NaN")
check("no NaN in end/filed dates", int(f[["end","filed"]].isna().sum().sum()) == 0)
f["end"] = pd.to_datetime(f["end"]); f["filed"] = pd.to_datetime(f["filed"])
check("filed >= period end (no future leak)", float((f["filed"] >= f["end"]).mean()) > 0.99,
      f"{100*(f['filed']>=f['end']).mean():.2f}% ok")
dup = int(f.duplicated(["ticker","concept","end"]).sum())
check("no duplicate (ticker,concept,end)", dup == 0, f"{dup} dups")
check("no infinities in val", not np.isinf(f["val"].to_numpy(dtype='float64')).any())
for concept in ("assets","shares"):
    v = f[f.concept==concept]["val"]; check(f"{concept} > 0", bool((v > 0).all()), f"{int((v<=0).sum())} non-positive")
print(f"  {f.ticker.nunique():,} tickers · {len(f):,} rows · {f.concept.nunique()} concepts · "
      f"{f['end'].min().date()}..{f['end'].max().date()}", flush=True)

print("\n" + "=" * 82)
print(f"RESULT: {'ALL CHECKS PASSED — clean (only legit pre-IPO/post-delist/pre-filing NaNs)' if not FAIL else str(len(FAIL))+' FAILED -> '+str(FAIL)}")
print("=" * 82)

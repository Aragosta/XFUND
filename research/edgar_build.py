#!/usr/bin/env python3
"""
edgar_build.py — (re)build the SEC EDGAR PIT fundamentals panel, WIDER than the current data/edgar/facts.parquet.

Adds: (a) more tickers — every price-universe ticker that maps to a SEC CIK (~5099 vs 4055 now);
      (b) NEW concepts — opinc (OperatingIncomeLoss = TRUE EBIT), sga, intexp, tax → EBIT/EV, ROIC, coverage;
      (c) the real `filed` date (exact point-in-time), alongside period `end`, so DATAHUB can lag correctly.

Output schema matches the reader (ticker, concept, end, val) + adds `filed`, `form`. Flows taken ANNUAL (fp=FY,
form 10-K) to match the existing convention (VALUE.py: "annual net income (10-K FY)"); balance-sheet stocks taken
as reported (latest-by-filed at each period end).

Env: LIMIT=<n> (test on first n tickers), OUT=<path> (default data/edgar/facts_v2.parquet).
SEC rules: ≤10 req/s, real User-Agent. Run:  python3 research/edgar_build.py
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import requests, pandas as pd, numpy as np
sys.path.insert(0, "/Users/enzokreeft/XFUND"); os.chdir("/Users/enzokreeft/XFUND")

UA = {"User-Agent": "XFUND research contact@xfund.example"}
OUT = os.environ.get("OUT", "data/edgar/facts_v2.parquet")
LIMIT = int(os.environ.get("LIMIT", 0))

# concept -> ordered us-gaap tag priority. STOCK = balance-sheet (instant); FLOW = income/cash (annual FY).
STOCK = {
    "book":        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "assets":      ["Assets"], "assets_cur": ["AssetsCurrent"], "liab_cur": ["LiabilitiesCurrent"],
    "cash":        ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "debt_lt":     ["LongTermDebtNoncurrent", "LongTermDebt"], "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent"],
    # NOTE: EntityCommonStockSharesOutstanding lives in the `dei` XBRL namespace, not `us-gaap` — the caller
    # must merge BOTH namespaces into the tag lookup dict (see main()/companyfacts()) or this silently returns
    # nothing for any filer that reports shares only via the dei cover-page tag (verified bug: ABT/ABBV/AAL — v1
    # facts.parquet had this data via a namespace-merging builder; my first us-gaap-only pass regressed it).
    "shares":      ["CommonStockSharesOutstanding", "dei:EntityCommonStockSharesOutstanding"],
}
FLOW = {
    "ni":   ["NetIncomeLoss"],
    "rev":  ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "ocf":  ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "dividends": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "rnd":  ["ResearchAndDevelopmentExpense"],
    # NEW concepts:
    "opinc":  ["OperatingIncomeLoss"],                                                    # TRUE EBIT
    "sga":    ["SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense",
               "SellingGeneralAndAdministrativeExpenses"],
    "intexp": ["InterestExpense", "InterestExpenseNonoperating"],
    "tax":    ["IncomeTaxExpenseBenefit"],
}


def _pick(units_by_tag, tags, annual):
    """Yield (end, filed, val, form) rows for the first present tag; annual→fp=FY only."""
    for tag in tags:
        arr = units_by_tag.get(tag)
        if not arr:
            continue
        out = []
        for e in arr:
            if "val" not in e or "end" not in e or "filed" not in e:
                continue
            if annual and e.get("fp") != "FY":                       # flows: keep annual FY only
                continue
            if annual and e.get("form", "").split("/")[0] not in ("10-K", "20-F", "40-F"):
                continue
            out.append((e["end"], e["filed"], float(e["val"]), e.get("form", "")))
        if out:
            return out
    return []


def main():
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=30)
    t2c = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in json.loads(r.text).values()}
    px = pd.read_parquet("data/daily.parquet")["close"]
    tickers = [t for t in px.columns if t.upper() in t2c]
    if LIMIT:
        tickers = tickers[:LIMIT]
    print(f"[edgar] {len(tickers)} price tickers map to a CIK; downloading companyfacts ...", flush=True)

    rows = []; sess = requests.Session(); sess.headers.update(UA); ok = 0; t0 = time.time()
    for i, tk in enumerate(tickers):
        cik = t2c[tk.upper()]
        try:
            resp = sess.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", timeout=30)
            if resp.status_code != 200:
                continue
            allfacts = json.loads(resp.text).get("facts", {})
            gaap = allfacts.get("us-gaap", {})
            dei = allfacts.get("dei", {})                        # EntityCommonStockSharesOutstanding lives here
            # collapse each tag to its USD (or shares) unit array; dei tags prefixed "dei:" to disambiguate
            ubt = {}
            for tag, node in gaap.items():
                units = node.get("units", {})
                ubt[tag] = units.get("USD") or units.get("shares") or next(iter(units.values()), [])
            for tag, node in dei.items():
                units = node.get("units", {})
                ubt[f"dei:{tag}"] = units.get("shares") or units.get("USD") or next(iter(units.values()), [])
            for concept, tags in {**STOCK, **FLOW}.items():
                for end, filed, val, form in _pick(ubt, tags, annual=(concept in FLOW)):
                    rows.append((tk, concept, end, filed, val, form))
            ok += 1
        except Exception:
            pass
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(tickers)}  ok={ok}  rows={len(rows)}  {(i+1)/(time.time()-t0):.1f} req/s", flush=True)
        time.sleep(0.11)                                             # ≤10 req/s SEC courtesy limit

    df = pd.DataFrame(rows, columns=["ticker", "concept", "end", "filed", "val", "form"])
    df["end"] = pd.to_datetime(df["end"]); df["filed"] = pd.to_datetime(df["filed"])
    # dedup: per (ticker, concept, end) keep the LATEST-filed value (as-restated superseded by PIT logic downstream)
    df = df.sort_values("filed").drop_duplicates(["ticker", "concept", "end"], keep="last")
    df.to_parquet(OUT)
    print(f"[edgar] wrote {OUT}: {len(df)} rows · {df.ticker.nunique()} tickers · concepts={sorted(df.concept.unique())}", flush=True)
    print(df.concept.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()

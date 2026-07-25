#!/usr/bin/env python3
"""build_sleeve_returns.py — generate MOM + MR deployable monthly net-return streams on the CURRENT (new) data,
for the ERC combine. DM is produced separately by DM.py (LONGONLY=1). Saves /tmp/mom_new.pkl, /tmp/mr_new.pkl."""
import warnings, os, sys, pickle
warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/Users/enzokreeft/XFUND"); sys.path.insert(0, "/Users/enzokreeft/XFUND/CURRENT BEST")
os.chdir("/Users/enzokreeft/XFUND")
import numpy as np, pandas as pd
from DATAHUB import DataHub

hub = DataHub(start="2000-01-01", min_days=0)     # ONE shared hub (new data) for both sleeves

# ── MOM: settled champion, long-only decile (deployable posture) ─────────────────────────────────────
from mom_layer import MomLayer
print("[build] MOM ...", flush=True)
mom = MomLayer(hub=hub, seeds=1, tier="liquid"); mom.build()
rMOM = mom.backtest(ls=False)                     # long-only decile, net
mom_ret = pd.Series(rMOM["returns"]); mom_ret.index = pd.DatetimeIndex(mom_ret.index)
pickle.dump(mom_ret.to_dict(), open("/tmp/mom_new.pkl", "wb"))
print(f"[build] MOM long-only net SR {rMOM['sharpe']:.2f}  rankIC {rMOM['rankIC']:.4f}  -> /tmp/mom_new.pkl", flush=True)

# ── MR: settled champion, daily book -> monthly returns ──────────────────────────────────────────────
from mr_layer import MrLayer
print("[build] MR ...", flush=True)
mr = MrLayer(hub=hub).build()
rMR = mr.backtest(cost=True, tag="MR")
mr_daily = pd.Series(rMR["returns"]); mr_daily.index = pd.DatetimeIndex(mr_daily.index)
mr_monthly = (1 + mr_daily).resample("ME").prod() - 1            # compound daily -> monthly
pickle.dump(mr_monthly.to_dict(), open("/tmp/mr_new.pkl", "wb"))
print(f"[build] MR net SR {rMR['sharpe']:.2f} (daily) -> monthly saved -> /tmp/mr_new.pkl", flush=True)
print("[build] done. run research/combine_erc.py after DM finishes.", flush=True)

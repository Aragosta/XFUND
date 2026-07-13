"""target_return.py — is the profitability target RETURN-RELEVANT? Learnable != tradeable. Measure IC to forward
returns for: cheapness (value), CURRENT profitability (Novy-Marx quality, tradeable), FUTURE profitability
(foresight CEILING — if you knew it, would you profit?), and IMPROVEMENT. Also the value x quality interaction.
Tells us how much of the return-relevant signal a profitability-forecasting model can actually capture."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub
hub=DataHub(); me,m_px,mret,elig=hub.me,hub.m_px,hub.mret,hub.elig("liquid"); f=hub.fund; mcap=hub.mcap()
gpa=((f("rev")-f("cogs"))/f("assets")).where(f("assets")>0); roe=(f("ni")/f("book")).where(f("book")>0); bm=f("book")/mcap
fwd1=mret.shift(-1); fwd6=m_px.shift(-6)/m_px-1
def rk(df): return df.rank(axis=1)                                              # cross-sectional rank (robust to fat tails)
sigs={
 "cheapness B/M (value)":      (rk(bm), "tradeable"),
 "current GP/A (quality)":     (rk(gpa), "tradeable"),
 "current ROE":                (rk(roe), "tradeable"),
 "improve GP/A (dGPA)":        (rk(gpa-gpa.shift(12)), "tradeable"),
 "improve ROE (dROE)":         (rk(roe-roe.shift(12)), "tradeable"),
 "cheap x quality":            (rk(bm)+rk(gpa), "tradeable"),
 "FUTURE GP/A (foresight)":    (rk(gpa.shift(-12)), "CEILING"),
 "FUTURE ROE (foresight)":     (rk(roe.shift(-12)), "CEILING"),
}
def ic(sig,fwd):
    cs=[]
    for d in me:
        x=sig.loc[d].where(elig.loc[d]); y=fwd.loc[d]; df=pd.DataFrame({"x":x,"y":y}).dropna()
        if len(df)<100: continue
        cs.append(df["x"].corr(df["y"],method="spearman"))
    a=np.array(cs); return np.nanmean(a), np.nanmean(a)/(np.nanstd(a)/np.sqrt(len(a))+1e-9)
print("="*80); print("IS THE PROFITABILITY TARGET RETURN-RELEVANT? IC to forward returns")
print(f"  {'signal':28}{'kind':10}{'IC(1m)':>9}{'t':>6}{'IC(6m)':>9}{'t':>6}")
for nm,(s,kind) in sigs.items():
    i1,t1=ic(s,fwd1); i6,t6=ic(s,fwd6)
    print(f"  {nm:28}{kind:10}{i1:>+9.3f}{t1:>+6.1f}{i6:>+9.3f}{t6:>+6.1f}")

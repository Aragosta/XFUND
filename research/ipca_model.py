#!/usr/bin/env python3
"""ipca_model.py — Instrumented Principal Component Analysis (Kelly, Pruitt & Su, JFE 2019).

WHY IPCA IS THE RIGHT NEXT STEP FROM OUR RIDGE. `shrinkage_combine.py` fits ONE STATIC weight vector over
five signal BOOKS: w ∝ (Σ+λI)⁻¹μ. That got SR 1.08 (vs best single 1.03, equal-weight 0.79, ERC 0.68), but it
is limited in two ways IPCA fixes:
  1. STATIC. The blend weights do not vary with the state of the market or with the stock. IPCA's loadings are
     a function of each stock's own characteristics, so exposure varies by NAME and by TIME.
  2. IT COMBINES BOOKS, NOT NAMES. Ridge operates on 5 portfolio return streams; IPCA operates on the full
     name-by-name panel and extracts LATENT FACTORS from it, which is where the dimension reduction lives.

THE MODEL
    r_{i,t+1} = β_{i,t}' f_{t+1} + ε_{i,t+1},      β_{i,t} = z_{i,t}' Γ            (restricted / "no alpha")
  z_{i,t}  L×1 observable characteristics of stock i at t (we use the momentum signal set + size + a constant)
  Γ        L×K matrix mapping characteristics → latent factor loadings  (the object being estimated)
  f_{t+1}  K×1 latent factors, estimated jointly

  Kelly-Pruitt-Su's point ("characteristics are covariances"): a characteristic earns a premium because it
  INSTRUMENTS a loading on a latent factor. Γ is L×K regardless of how many stocks there are, so the model
  ingests a large cross-section without the overfitting that plain PCA suffers.

ESTIMATION — alternating least squares to convergence:
    given Γ:  f_t = (Γ'Z_t'Z_t Γ)⁻¹ Γ'Z_t' r_t                       (cross-sectional GLS each month)
    given f:  vec(Γ) = [Σ_t (Z_t'Z_t) ⊗ (f_t f_t')]⁻¹ vec(Σ_t Z_t' r_t f_t')
  then normalise Γ to be orthonormal (Γ'Γ = I) and fix the sign so each factor has a positive mean — the
  standard identification, since (Γ, f) is only identified up to a K×K rotation.

PREDICTION (leak-free): fit on months strictly before t, then
    E[r_{i,t+1}] = z_{i,t}' Γ̂ f̄,     f̄ = mean of the estimated factors over the TRAINING window.
Refit annually on an EXPANDING window, exactly like DM.py, so nothing from the future touches a forecast.

Books are then built through the identical construction used for every other arm today (Q3 size bucket,
hold=6, banded, measured IBKR costs) so the comparison against ridge / best-single is apples-to-apples.
"""
import warnings, sys, os
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, BACKTEST
from horse_race_v2 import S, tc, bf, hub, pnl, spy, adv, parts
from dm_capacity import book

CHARS = ["hi52", "mom11", "tvalpast", "resmom", "fip"]
MAXIT, TOL = 200, 1e-6


def _xs_standardise(v):
    """Cross-sectional rank map to [-0.5, 0.5] — KPS standardise characteristics each period."""
    r = pd.Series(v).rank(pct=True).values
    return r - 0.5


def build_panel():
    """→ dict month → (Z [N×L], r [N], idx). Z columns = CHARS + size + constant, all known at t."""
    dates = sorted(set.intersection(*[set(S[c]) for c in CHARS]))
    out = {}
    for d in dates:
        idx = S[CHARS[0]][d].dropna().index
        for c in CHARS[1:]:
            idx = idx.intersection(S[c][d].dropna().index)
        if len(idx) < 200 or d not in adv.index: continue
        Z = np.column_stack([_xs_standardise(S[c][d].reindex(idx).values) for c in CHARS])
        sz = adv.loc[d].reindex(idx)
        Z = np.column_stack([Z, _xs_standardise(np.log(sz.clip(lower=1)).values), np.ones(len(idx))])
        ok = np.isfinite(Z).all(axis=1)
        out[d] = (Z[ok], idx[ok])
    return out, CHARS + ["size", "const"]


def fit_ipca(Zs, rs, K, maxit=MAXIT, tol=TOL, seed=0, ridge=0.0):
    """Alternating least squares. Zs/rs are lists of per-month (Z, r). Returns Γ (L×K) and F (T×K)."""
    L = Zs[0].shape[1]
    rng = np.random.default_rng(seed)
    G = np.linalg.qr(rng.standard_normal((L, K)))[0]
    F = np.zeros((len(Zs), K))
    for it in range(maxit):
        # --- factors given Γ (cross-sectional GLS each month) ---
        for t, (Z, r) in enumerate(zip(Zs, rs)):
            B = Z @ G                                       # N×K instrumented loadings
            A = B.T @ B
            F[t] = np.linalg.solve(A + 1e-10 * np.eye(K), B.T @ r) if np.linalg.cond(A) < 1e12 else 0.0
        # --- Γ given factors (stacked normal equations, Kronecker form) ---
        M = np.zeros((L * K, L * K)); v = np.zeros(L * K)
        for t, (Z, r) in enumerate(zip(Zs, rs)):
            ZtZ = Z.T @ Z; ff = np.outer(F[t], F[t])
            # vec(AXB) = (Bᵀ ⊗ A)vec(X). The FOC is Σ Z'Z Γ ff' = Σ Z'r f', so the Kronecker is
            # (ff' ⊗ Z'Z), NOT (Z'Z ⊗ ff'), and the RHS is vec(Z'r f') = kron(f, Z'r). And vec is
            # COLUMN-major, so the solution must be reshaped with order="F". All three were wrong and
            # together they produced a Γ that was anti-predictive even IN SAMPLE (rank-IC -0.043).
            M += np.kron(ff, ZtZ)
            v += np.kron(F[t], Z.T @ r)
        # SHRINKAGE ON Γ. Without it, IPCA on collinear characteristics reproduces the unregularised
        # multivariate fit, whose coefficients sign-flip against their own univariate ICs (hi52: +0.069 IC
        # but -0.011 pooled-OLS coefficient) because N_eff=1.80 leaves the design near-singular. Kozak-
        # Nagel-Santosh's whole point: dense estimation over correlated characteristics REQUIRES shrinkage.
        pen = ridge * np.trace(M) / max(L * K, 1)
        Gn = np.linalg.solve(M + (pen + 1e-8) * np.eye(L * K), v).reshape(L, K, order="F")
        Q, R = np.linalg.qr(Gn)                             # identification: Γ'Γ = I
        Q = Q * np.sign(np.diag(R))                         # BUG FIX: QR is unique only up to the sign of
        Gn = Q                                              # diag(R); without this the columns of Γ can flip
        if np.max(np.abs(Gn - G)) < tol:                    # sign between iterations while F does not.
            G = Gn; break
        G = Gn
    # BUG FIX: F above was estimated against the PREVIOUS G. Re-estimate it against the FINAL G, otherwise
    # Γ and f̄ are one update out of step and the prediction z'Γf̄ can carry the wrong sign on a factor.
    for t, (Z, r) in enumerate(zip(Zs, rs)):
        B = Z @ G; A = B.T @ B
        F[t] = np.linalg.solve(A + 1e-10 * np.eye(K), B.T @ r) if np.linalg.cond(A) < 1e12 else 0.0
    for k in range(K):                                      # sign convention: positive mean factor
        if F[:, k].mean() < 0: F[:, k] *= -1; G[:, k] *= -1
    return G, F


def ipca_scores(K=3, first_year=2011, min_train=60, RIDGE=0.0):
    panel, names = build_panel()
    dates = sorted(panel)
    out = {}
    cur = None
    for i, d in enumerate(dates):
        if d.year < first_year or i < min_train: continue
        if cur != d.year:                                   # ANNUAL refit, EXPANDING window
            cur = d.year
            tr = dates[:i]                                  # strictly past
            Zs, rs = [], []
            for j, dt in enumerate(tr[:-1]):
                Z, idx = panel[dt]
                nxt = dates[tr.index(dt) + 1]
                fwd = hub.mret.reindex(index=[nxt.to_timestamp("M")]).iloc[0] if hasattr(nxt, "to_timestamp") \
                    else hub.mret.loc[nxt]
                r = fwd.reindex(idx).values
                m = np.isfinite(r)
                if m.sum() < 100: continue
                Zs.append(Z[m]); rs.append(np.nan_to_num(r[m]))
            if len(Zs) < 24: continue
            G, F = fit_ipca(Zs, rs, K, ridge=RIDGE)
            fbar = F.mean(axis=0)
        Z, idx = panel[d]
        out[d] = pd.Series(Z @ G @ fbar, index=idx)         # E[r] = z'Γ f̄
    return out


def stats(tag, x):
    D = pd.concat({"r": x, "m": spy}, axis=1).dropna()
    if len(D) < 24: return
    X = np.c_[np.ones(len(D)), D.m.values]; c, *_ = np.linalg.lstsq(X, D.r.values, rcond=None)
    e = D.r.values - X @ c
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(D) - 2, 1)))
    eq = (1 + D.r).cumprod()
    print(f"  {tag:30}{D.r.mean()*12:>8.1%}{D.r.std()*np.sqrt(12):>7.1%}"
          f"{D.r.mean()*12/(D.r.std()*np.sqrt(12)+1e-9):>6.2f}{(eq/eq.cummax()-1).min():>8.1%}"
          f"{c[1]:>6.2f}{c[0]*12:>+9.2%}{c[0]/(se[0]+1e-12):>6.2f}", flush=True)


if __name__ == "__main__":
    q = adv.rank(axis=1, pct=True); Q3 = (q > 0.4) & (q <= 0.6)
    print(f"\n[IPCA] Kelly-Pruitt-Su · chars = {CHARS} + size + const · Q3 · hold=6 · band · IBKR costs")
    H = f"  {'model':30}{'ann':>8}{'vol':>7}{'SR':>6}{'maxDD':>8}{'beta':>6}{'alpha':>9}{'t':>6}"
    print(H)
    for K, RG in [(k, r) for k in (1, 3) for r in (0.0, 0.1, 1.0, 10.0, 100.0)]:
        try:
            Sc = ipca_scores(K=K, RIDGE=RG)
            if not Sc: print(f"  IPCA K={K}: no scores"); continue
            W = book(Sc, uni=Q3)
            r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index),
                                  transaction_cost=tc, borrow_fee=bf)
            x = r["returns"].dropna(); x = x[x.index >= W.index.min()]
            x.index = pd.PeriodIndex(x.index, freq="M")
            stats(f"IPCA K={K} ridge={RG:g}", x)
        except Exception as ex:
            print(f"  IPCA K={K} r={RG}: {type(ex).__name__}: {ex}")
    print("\n  benchmarks from earlier today: best single mom11 SR 1.03 · ridge SR 1.08 ·"
          " equal-weight 0.79 · ERC 0.68")

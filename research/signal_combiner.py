#!/usr/bin/env python3
"""signal_combiner.py — THE reusable multi-signal combiner: add any signal, now or later.

DESIGN REQUIREMENT (user, 2026-07-25): "a model where we can add all momentum signals, and new ones when
we find them." Everything below follows from what was measured today rather than from taste:

  1. IT MUST SHRINK, OR IT INVERTS. Unregularised joint estimation over correlated signals sign-flips every
     input against its own univariate IC (hi52: univariate rank-IC +0.069, pooled-OLS coefficient -0.011;
     Fama-MacBeth agrees; IPCA reproduces it). Any "throw all the signals in" model without shrinkage gets
     WORSE as you add signals, because new collinear columns load onto near-null directions.
  2. THE GAIN IS CAPPED BY EFFECTIVE BREADTH. N_eff = (Σλ)²/Σλ² on the book-correlation matrix was 1.80 of
     5 nominal signals. Grinold: IR ≈ IC·√breadth, so five 0.85-correlated signals buy √1.80 = 1.34x, not
     √5 = 2.24x. Adding a SIXTH momentum variant moves N_eff by ~0.05. Breadth comes from new DATA, not
     new arithmetic on the same price path.
  3. SHRINK THE EIGENVALUES, NOT ISOTROPICALLY. The book-correlation eigenvalues were [3.64, .70, .35, .22,
     .10]. Isotropic λI penalises the 3.64 direction (well estimated) exactly as hard as the 0.10 direction
     (pure noise). Kozak-Nagel-Santosh shrink in PC space, proportional to 1/eigenvalue — implemented here.
  4. SHRINK THE MEANS TOO. We regularised Σ but left μ raw, and means are the noisier object. James-Stein
     shrinks each signal's mean toward the grand mean by a factor driven by its own estimation error.
  5. JUDGE ON THE BOOK, NEVER THE SCORE. T31 measured score-correlation 0.15 and called a signal orthogonal;
     T33 measured book-correlation 0.74 for the same idea. Construction re-correlates books that looked
     independent as scores. Every statistic here is computed on NET BOOK RETURNS.

WHAT IT PROVIDES
    SignalCombiner.add(name, scores)     register a signal (dict date -> Series of per-name scores)
    .fit_weights(method, ...)            expanding-window, leak-free allocator weights
    .combined()                          the combined net return stream
    .admit(name, scores)                 THE ADMISSION TEST for a candidate signal — the answer to
                                         "should this new signal go in?" It is NOT "is its Sharpe good".
                                         Reports: ΔN_eff · book-corr to the current composite · spanning
                                         alpha BOTH ways (new | composite, and composite | new).
                                         A 0.3-Sharpe signal at 0.2 correlation beats a 0.6-Sharpe signal
                                         at 0.95 correlation; this makes that tradeoff explicit.

Benchmarks to beat, all measured today on Q3 / hold=6 / banded / measured IBKR costs:
    best single (mom11) SR 1.03 · isotropic ridge SR 1.08 · equal-weight 0.79 · ERC 0.68 · IPCA negative
"""
import warnings, sys, os
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, BACKTEST


def n_eff(R: pd.DataFrame) -> float:
    """Effective number of INDEPENDENT signals: (Σλ)²/Σλ² on the correlation matrix of book returns."""
    ev = np.linalg.eigvalsh(R.corr().values)
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def james_stein(mu: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Shrink each signal's mean toward the grand mean; more shrinkage when its own SE is large."""
    gm = mu.mean(); d = mu - gm
    s2 = (se ** 2).mean()
    denom = float((d ** 2).sum())
    if denom <= 0 or len(mu) < 3: return mu
    c = max(0.0, 1.0 - (len(mu) - 2) * s2 / denom)
    return gm + c * d


def kns_weights(H: pd.DataFrame, kappa: float = 1.0, js: bool = True) -> np.ndarray:
    """Kozak-Nagel-Santosh style shrunk tangency weights on the signal return panel `H` (past only).

    Rotate into the PC basis of Σ, then shrink each PC's contribution by λ/(λ + κ·λ̄). A high-eigenvalue
    (well-estimated) direction is barely touched; a near-null direction is crushed. Isotropic ridge cannot
    make that distinction, which is why it leaves return on the table when eigenvalues are spread.
    """
    mu = H.mean().values
    if js:
        se = H.std().values / np.sqrt(max(len(H), 1))
        mu = james_stein(mu, se)
    S = np.cov(H.values, rowvar=False)
    ev, V = np.linalg.eigh(S)
    ev = np.clip(ev, 1e-12, None)
    shrunk = ev + kappa * ev.mean()                       # PC-proportional shrinkage
    w = V @ np.diag(1.0 / shrunk) @ V.T @ mu
    n = np.abs(w).sum()
    return w / n if n > 0 else np.full(len(mu), 1.0 / len(mu))


class SignalCombiner:
    def __init__(self, builder, backtester):
        """builder(scores) -> weight DataFrame ; backtester(W) -> net monthly return Series."""
        self.builder, self.backtester = builder, backtester
        self.books, self.rets = {}, {}

    def add(self, name, scores):
        W = self.builder(scores)
        self.books[name] = W
        self.rets[name] = self.backtester(W)
        return self

    def panel(self):
        return pd.DataFrame(self.rets).dropna()

    def fit_weights(self, method="kns", win=36, kappa=1.0):
        R = self.panel(); out = pd.DataFrame(0.0, index=R.index, columns=R.columns)
        for i in range(win, len(R)):
            H = R.iloc[:i]                                # strictly past — leak-free
            if method == "kns":     w = kns_weights(H, kappa)
            elif method == "ridge": w = np.linalg.solve(np.cov(H.values, rowvar=False)
                                                        + kappa * np.eye(R.shape[1]), H.mean().values)
            else:                   w = np.full(R.shape[1], 1.0 / R.shape[1])
            s = np.abs(w).sum()
            out.iloc[i] = w / s if s > 0 else 0.0
        return out

    def combined(self, method="kns", win=36, kappa=1.0):
        R = self.panel(); W = self.fit_weights(method, win, kappa)
        x = (R * W.values).sum(axis=1)
        return x[x != 0]

    # ── THE ADMISSION TEST ────────────────────────────────────────────────────
    def admit(self, name, scores, method="kns", kappa=1.0):
        """Should this candidate signal join the library? Judged on INDEPENDENCE, not standalone Sharpe."""
        base_R = self.panel(); base = self.combined(method, kappa=kappa)
        Wc = self.builder(scores); rc = self.backtester(Wc)
        both = pd.concat({"cand": rc, "base": base}, axis=1).dropna()
        if len(both) < 36:
            print(f"  {name}: too short to judge ({len(both)} months)"); return None
        n0 = n_eff(base_R)
        n1 = n_eff(pd.concat([base_R, rc.rename(name)], axis=1).dropna())
        corr = both["cand"].corr(both["base"])

        def span(y, x):
            X = np.c_[np.ones(len(x)), x.values]
            c, *_ = np.linalg.lstsq(X, y.values, rcond=None)
            e = y.values - X @ c
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(x) - 2, 1)))
            return c[0] * 12, c[0] / (se[0] + 1e-12)

        a_new, t_new = span(both["cand"], both["base"])    # does the candidate survive the composite?
        a_old, t_old = span(both["base"], both["cand"])    # does the composite survive the candidate?
        sr = both["cand"].mean() * 12 / (both["cand"].std() * np.sqrt(12) + 1e-9)
        verdict = ("ADMIT" if (t_new > 2.0 and n1 - n0 > 0.15) else
                   "MARGINAL" if (t_new > 1.5 or n1 - n0 > 0.10) else "REJECT")
        print(f"  {name:16} SR {sr:>5.2f} · corr→composite {corr:>+5.2f} · N_eff {n0:.2f}→{n1:.2f} "
              f"(Δ{n1-n0:+.2f}) · α|comp {a_new:>+7.2%} (t={t_new:>5.2f}) · comp|α {a_old:>+7.2%} "
              f"(t={t_old:>5.2f}) → {verdict}", flush=True)
        return dict(name=name, sr=sr, corr=corr, dn=n1 - n0, alpha=a_new, t=t_new, verdict=verdict)


if __name__ == "__main__":
    from horse_race_v2 import S, tc, bf, pnl, adv, spy
    from dm_capacity import book as _book
    q = adv.rank(axis=1, pct=True); Q3 = (q > 0.4) & (q <= 0.6)

    def builder(sc): return _book(sc, uni=Q3)

    def bt(W):
        r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index),
                              transaction_cost=tc, borrow_fee=bf)
        x = r["returns"].dropna(); x = x[x.index >= W.index.min()]
        x.index = pd.PeriodIndex(x.index, freq="M"); return x

    def stats(tag, x):
        D = pd.concat({"r": x, "m": spy}, axis=1).dropna()
        X = np.c_[np.ones(len(D)), D.m.values]; c, *_ = np.linalg.lstsq(X, D.r.values, rcond=None)
        e = D.r.values - X @ c
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(D) - 2, 1)))
        eq = (1 + D.r).cumprod()
        print(f"  {tag:28}{D.r.mean()*12:>8.1%}{D.r.std()*np.sqrt(12):>7.1%}"
              f"{D.r.mean()*12/(D.r.std()*np.sqrt(12)+1e-9):>6.2f}{(eq/eq.cummax()-1).min():>8.1%}"
              f"{c[0]*12:>+9.2%}{c[0]/(se[0]+1e-12):>6.2f}", flush=True)

    core = ["hi52", "mom11", "tvalpast", "resmom"]
    C = SignalCombiner(builder, bt)
    for n in core: C.add(n, S[n])
    R = C.panel()
    print(f"\n[signal_combiner] {len(R)} months · {len(core)} signals · N_eff = {n_eff(R):.2f}")
    print(f"\n{'combiner':30}{'ann':>8}{'vol':>7}{'SR':>6}{'maxDD':>8}{'alpha':>9}{'t':>6}")
    for n in core: stats(f"single: {n}", R[n])
    stats("equal-weight", C.combined("equal"))
    for k in (0.1, 1.0, 10.0): stats(f"isotropic ridge κ={k:g}", C.combined("ridge", kappa=k))
    for k in (0.1, 0.5, 1.0, 5.0): stats(f"KNS shrinkage κ={k:g}", C.combined("kns", kappa=k))

    print(f"\nADMISSION TEST — should these join the library?")
    C.admit("fip", S["fip"])
    C.admit("composite", S["composite"])

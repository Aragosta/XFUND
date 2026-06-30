"""
execution_layer.py — DynTrad (Gârleanu–Pedersen 2013) execution layer.

Implements the full optimal dynamic trading policy from:
    "Dynamic Trading with Predictable Returns and Transaction Costs"
    Gârleanu & Pedersen, Journal of Finance, 2013.

The key insight: under quadratic transaction costs and mean-reverting signals,
the optimal policy is to (1) aim in front of the target and (2) trade partially
toward the aim. Signals with slower mean reversion get more weight in the aim
portfolio because their alpha persists longer and is worth more patient trading.

Public API
----------
    DynTrad(signal_decay, risk_aversion, cost_matrix, cov_matrix, ...)
        .step(signals, current_position) → new_position
        .run(signal_history)             → full position history

    estimate_signal_decay(signals)       → per-signal φ vector
    estimate_cost_matrix(dollar_volume)   → diagonal Λ from liquidity
"""

import numpy as np
import pandas as pd


def estimate_signal_decay(signals: pd.DataFrame, method: str = "acf1") -> np.ndarray:
    """
    Estimate per-signal mean-reversion rate φ ∈ (0, 1).

    φ close to 1 → persistent signal (slow decay, aim further ahead).
    φ close to 0 → fast-decaying signal (trade aggressively now).

    For a single cross-sectional signal (columns = assets), this estimates
    the average autocorrelation across assets. For multiple named signals
    (columns = signal names), it estimates per-signal persistence.
    """
    if method == "acf1":
        phi = np.zeros(signals.shape[1])
        for j in range(signals.shape[1]):
            col = signals.iloc[:, j].dropna()
            if len(col) < 3:
                phi[j] = 0.5
                continue
            col_dm = col - col.mean()
            c0 = (col_dm ** 2).mean()
            if c0 < 1e-15:
                phi[j] = 0.0
                continue
            c1 = (col_dm.iloc[:-1].values * col_dm.iloc[1:].values).mean()
            phi[j] = np.clip(c1 / c0, 0.01, 0.99)
        return phi
    raise ValueError(f"Unknown method: {method}")


def estimate_cost_matrix(
    dollar_volume: pd.DataFrame,
    base_cost_bps: float = 10.0,
    impact_exponent: float = -0.5,
) -> pd.DataFrame:
    """
    Estimate per-asset quadratic cost coefficient Λ_ii from dollar volume.

    Under the Gârleanu-Pedersen model, TC(Δx) = ½ Δx' Λ Δx (quadratic).
    Λ_ii scales inversely with liquidity: less liquid → higher cost.

    Returns a (dates × tickers) DataFrame of per-asset cost coefficients,
    compatible with the tiered_transaction_costs from BACKTEST.py but
    expressed as the quadratic cost coefficient rather than a flat rate.

    Λ_ii ≈ base_cost * (median_dv / dv_i)^|exponent|
    """
    base = base_cost_bps / 1e4
    median_dv = dollar_volume.median(axis=1)
    ratio = dollar_volume.div(median_dv, axis=0).clip(lower=0.01)
    cost = base * ratio.pow(impact_exponent)
    return cost.fillna(cost.max().max())


class DynTrad:
    """
    Full Gârleanu–Pedersen (2013) optimal dynamic trading policy.

    Parameters
    ----------
    signal_decay : array-like, shape (K,)
        Per-signal mean-reversion rate φ_k ∈ (0, 1). Higher = more persistent.
        For a single cross-sectional signal, K=1 and φ is the average
        autocorrelation of the signal across assets.

    risk_aversion : float
        γ — risk aversion coefficient. Higher = smaller positions.

    cost_multiplier : float
        λ — scales the transaction cost matrix Λ. Higher = trade slower.

    discount_rate : float
        ρ — time discount rate per period. Typically small (e.g. 0.01/12
        for monthly with 1% annual discount).

    cov_matrix : ndarray, shape (N, N), optional
        Σ — asset return covariance matrix. If None, uses identity
        (equivalent to assuming uncorrelated, equal-variance assets).

    gross_exposure : float
        Target gross exposure (sum of |weights|). Set to 2.0 for
        dollar-neutral long-short.
    """

    def __init__(
        self,
        signal_decay: np.ndarray,
        risk_aversion: float = 1.0,
        cost_multiplier: float = 1.0,
        discount_rate: float = 0.001,
        cov_matrix: np.ndarray | None = None,
        gross_exposure: float = 2.0,
    ):
        self.phi = np.atleast_1d(np.asarray(signal_decay, dtype=float))
        self.gamma = risk_aversion
        self.lam = cost_multiplier
        self.rho = discount_rate
        self.cov = cov_matrix
        self.gross = gross_exposure

        self.rate = self._compute_trading_rate()
        self.aim_weights = self._compute_aim_weights()

    def _compute_trading_rate(self) -> float:
        """
        Optimal trading rate from GP Proposition 2.

        a = [-(γ + λρ) + √((γ + λρ)² + 4γλ(1-ρ))] / [2(1-ρ)λ]

        This is the fraction of the gap between current position and aim
        that should be traded each period. Derived from the Bellman equation
        under quadratic costs.
        """
        g, l, r = self.gamma, self.lam, self.rho
        b = g + l * r
        discriminant = b ** 2 + 4 * g * l * (1 - r)
        a = (-b + np.sqrt(discriminant)) / (2 * (1 - r) * l)
        return float(a)

    def _compute_aim_weights(self) -> np.ndarray:
        """
        Per-signal aim weights from GP Proposition 2.

        aim_weight_k = 1 / (1 + φ_k * a / γ)

        Signals with higher persistence (φ close to 1) get aim_weight < 1,
        meaning the aim portfolio discounts their current value because
        the signal will still be there next period. Fast-decaying signals
        (φ close to 0) get aim_weight ≈ 1 — trade toward them now.
        """
        return 1.0 / (1.0 + self.phi * self.rate / self.gamma)

    @property
    def effective_delta(self) -> float:
        """
        The effective partial-adjustment rate δ = a / λ.
        Comparable to the old hardcoded δ parameter.
        """
        return self.rate / self.lam

    def markowitz(self, signals: np.ndarray) -> np.ndarray:
        """
        Markowitz portfolio: (γΣ)⁻¹ · signals.

        For a single signal with identity covariance, this is just
        signals / γ. With a real covariance matrix, it optimally
        diversifies across correlated assets.
        """
        if self.cov is not None:
            return np.linalg.solve(self.gamma * self.cov, signals)
        return signals / self.gamma

    def aim(self, signals: np.ndarray) -> np.ndarray:
        """
        Aim portfolio: the Markowitz portfolio with signals scaled by
        the per-signal aim weights.

        aim_t = Markowitz(w_1 * f_1, ..., w_K * f_K)

        For a single cross-sectional signal (the common case), this
        scales all signals uniformly by the single aim weight.
        """
        if len(self.aim_weights) == 1:
            scaled = signals * self.aim_weights[0]
        else:
            scaled = signals * self.aim_weights
        return self.markowitz(scaled)

    def step(
        self,
        signals: np.ndarray,
        current_position: np.ndarray,
    ) -> np.ndarray:
        """
        One step of the optimal DynTrad policy.

        x_t = (1 - δ) · x_{t-1} + δ · aim_t

        where δ = a/λ (the effective trading rate) and aim_t is the
        aim portfolio computed from current signals.

        Returns the new position, normalized to target gross exposure.
        """
        delta = self.effective_delta
        aim_t = self.aim(signals)
        new_pos = (1.0 - delta) * current_position + delta * aim_t

        gross = np.abs(new_pos).sum()
        if gross > 0:
            new_pos = new_pos * (self.gross / gross)

        return new_pos

    def run(
        self,
        signal_df: pd.DataFrame,
        cost_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Run DynTrad over a full signal history.

        Parameters
        ----------
        signal_df : DataFrame (dates × assets)
            Target signals / raw weights at each rebalance date.
        cost_df : DataFrame (dates × assets), optional
            Per-asset cost coefficients. If provided, overrides
            self.lam with per-asset scaling (Λ_ii).

        Returns
        -------
        DataFrame (dates × assets) of optimally-adjusted positions.
        """
        cols = signal_df.columns
        dates = signal_df.index
        positions = {}
        w_prev = np.zeros(len(cols))

        for date in dates:
            signals = signal_df.loc[date].values.astype(float)
            signals = np.nan_to_num(signals, 0.0)

            if cost_df is not None and date in cost_df.index:
                asset_costs = cost_df.loc[date].reindex(cols).fillna(
                    cost_df.loc[date].max()
                ).values
                per_asset_lam = asset_costs / np.median(
                    asset_costs[asset_costs > 0]
                ) if np.any(asset_costs > 0) else np.ones(len(cols))
            else:
                per_asset_lam = np.ones(len(cols))

            delta = self.effective_delta
            aim_t = self.aim(signals)
            scaled_aim = aim_t / np.clip(per_asset_lam, 0.1, 10.0)

            new_pos = (1.0 - delta) * w_prev + delta * scaled_aim

            gross = np.abs(new_pos).sum()
            if gross > 0:
                new_pos = new_pos * (self.gross / gross)

            positions[date] = new_pos
            w_prev = new_pos

        return pd.DataFrame(positions, index=cols).T

    def summary(self) -> dict:
        """Return key parameters for inspection / logging."""
        return {
            "signal_decay_phi": self.phi.tolist(),
            "risk_aversion_gamma": self.gamma,
            "cost_multiplier_lambda": self.lam,
            "discount_rate_rho": self.rho,
            "trading_rate_a": self.rate,
            "effective_delta": self.effective_delta,
            "aim_weights": self.aim_weights.tolist(),
            "gross_exposure": self.gross,
        }


def run_dyntrad(
    raw_weights: pd.DataFrame,
    signal_decay: float | np.ndarray = 0.65,
    risk_aversion: float = 1.0,
    cost_multiplier: float = 1.0,
    discount_rate: float = 0.001,
    gross_exposure: float = 2.0,
    dollar_volume: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Convenience wrapper: apply DynTrad to a raw weight matrix.

    Parameters
    ----------
    raw_weights : DataFrame (dates × assets)
        Target portfolio weights from the signal model.
    signal_decay : float or array
        φ — signal persistence. 0.65 is the measured autocorrelation
        of the DM-RET signal in XFUND.
    risk_aversion : float
        γ — higher = smaller positions.
    cost_multiplier : float
        λ — higher = trade slower.
    discount_rate : float
        ρ — time discount per period.
    gross_exposure : float
        Target |w|₁.
    dollar_volume : DataFrame, optional
        If provided, estimates per-asset cost matrix from liquidity.

    Returns
    -------
    (adjusted_weights, params_dict)
    """
    phi = np.atleast_1d(np.asarray(signal_decay, dtype=float))

    cost_df = None
    if dollar_volume is not None:
        cost_df = estimate_cost_matrix(dollar_volume)

    engine = DynTrad(
        signal_decay=phi,
        risk_aversion=risk_aversion,
        cost_multiplier=cost_multiplier,
        discount_rate=discount_rate,
        gross_exposure=gross_exposure,
    )

    adjusted = engine.run(raw_weights, cost_df=cost_df)
    return adjusted, engine.summary()


# ═══════════════════════════════════════════════════════════════════════════════
# ## META LOSS
#
# Gradient-based fine-tuning of DynTrad parameters toward realized Sharpe.
#
# Architecture:
#     signals → DynTrad(γ, λ, φ) → weights → returns - costs → Sharpe
#                    ↑                                          |
#                    └──── gradient descent ←────────────────────┘
#
# The DynTrad update rule  x_t = (1-δ)x_{t-1} + δ·aim_t  is fully
# differentiable. The meta-loss backpropagates through the entire
# position trajectory to optimize (γ, λ, φ) for net Sharpe.
#
# Key design choices (TODO):
#   - PyTorch implementation of DynTrad.step() for autograd
#   - Loss = -Sharpe(net_returns) with L2 regularization on parameter
#     deviation from the analytical GP starting point
#   - Expanding-window walk-forward to prevent overfitting on 180 months
#   - Parameter space is tiny (3-5 scalars) → overfitting is manageable
#   - Initial parameters from the analytical GP closed form, meta-loss
#     fine-tunes from there
#
# NOT YET IMPLEMENTED.
# ═══════════════════════════════════════════════════════════════════════════════

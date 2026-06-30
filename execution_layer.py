"""
execution_layer.py — DynTrad (Gârleanu–Pedersen 2013) execution layer.

Implements the optimal dynamic trading policy from:
    "Dynamic Trading with Predictable Returns and Transaction Costs"
    Gârleanu & Pedersen, Journal of Finance, 2013.

The key insight: under quadratic transaction costs and mean-reverting signals,
the optimal policy is to (1) aim in front of the target and (2) trade partially
toward the aim. Signals with slower mean reversion get more weight in the aim
portfolio because their alpha persists longer and is worth more patient trading.

Two modes of operation:

  "weights" mode (default, XFUND use case):
      Input = target portfolio weights (already in position space).
      DynTrad smooths toward these weights using the optimal trading rate.
      For a single signal, aim weights cancel after gross normalization,
      so only the trading fraction a/λ matters.

  "signals" mode (paper's original setup):
      Input = raw return-predicting factors f.
      DynTrad applies (γΣ)⁻¹ B to convert to Markowitz positions,
      scales by per-signal aim weights, then smooths.

Public API
----------
    DynTrad(signal_decay, risk_aversion, cost_multiplier, ...)
        .step(target, current_position) → new_position
        .run(signal_history)             → full position history

    estimate_signal_decay(signals)       → per-signal φ vector
    estimate_cost_matrix(dollar_volume)  → diagonal Λ from liquidity
"""

import numpy as np
import pandas as pd


def estimate_signal_decay(signals: pd.DataFrame, method: str = "acf1") -> np.ndarray:
    """
    Estimate per-signal mean-reversion rate φ ∈ (0, 1).

    φ close to 1 → persistent signal (slow decay, aim further ahead).
    φ close to 0 → fast-decaying signal (trade aggressively now).
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

    Under the GP model, TC(Δx) = ½ Δx' Λ Δx (quadratic).
    Λ_ii scales inversely with liquidity: less liquid → higher cost.
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
        Per-signal mean-reversion rate φ_k ∈ (0, 1).

    risk_aversion : float
        γ — risk aversion coefficient. Under Assumption A (Λ = λΣ),
        higher γ → faster trading (risk of deviating from aim hurts more).

    cost_multiplier : float
        λ — the scalar in Λ = λΣ. Higher → trade slower.
        In the paper's empirical application, λ = 2. For XFUND with
        ~10bp costs and ~3.3% monthly vol, λ ≈ 2 is reasonable.

    discount_rate : float
        ρ ∈ (0, 1) — time discount rate per period.

    input_type : str
        "weights" — input is target portfolio weights (XFUND default).
                    Markowitz conversion is skipped.
        "signals" — input is raw return-predicting factors f.
                    Aim applies (γΣ)⁻¹ conversion per the paper.

    cov_matrix : ndarray, optional
        Σ — only used in "signals" mode for the Markowitz conversion.

    gross_exposure : float
        Target gross exposure (sum of |weights|).
    """

    def __init__(
        self,
        signal_decay: np.ndarray,
        risk_aversion: float = 1.0,
        cost_multiplier: float = 2.0,
        discount_rate: float = 0.001,
        input_type: str = "weights",
        cov_matrix: np.ndarray | None = None,
        gross_exposure: float = 2.0,
    ):
        self.phi = np.atleast_1d(np.asarray(signal_decay, dtype=float))
        self.gamma = risk_aversion
        self.lam = cost_multiplier
        self.rho = discount_rate
        self.input_type = input_type
        self.cov = cov_matrix
        self.gross = gross_exposure

        self._a = self._compute_a()
        self._trading_frac = np.clip(self._a / self.lam, 0.0, 1.0)
        self.aim_weights = self._compute_aim_weights()

    def _compute_a(self) -> float:
        """
        Solve for a from GP Proposition 2, equation (9).

        a = [-(γ + λρ) + √((γ + λρ)² + 4γλ(1-ρ))] / [2(1-ρ)λ]

        Under Assumption A (Λ = λΣ), the trading fraction a/λ < 1.
        """
        g, l, r = self.gamma, self.lam, self.rho
        b = g + l * r
        discriminant = b ** 2 + 4 * g * l * (1 - r)
        return float((-b + np.sqrt(discriminant)) / (2 * (1 - r) * l))

    def _compute_aim_weights(self) -> np.ndarray:
        """
        Per-signal aim weights from GP Proposition 4, equation (15).

        aim_weight_k = 1 / (1 + φ_k · a / γ)

        Signals with higher persistence get lower aim weights — the
        investor discounts them because they'll persist into future periods.
        """
        return 1.0 / (1.0 + self.phi * self._a / self.gamma)

    @property
    def trading_fraction(self) -> float:
        """
        The trading fraction δ = a/λ ∈ (0, 1].

        From equation (10): x_t = (1-δ) x_{t-1} + δ aim_t
        """
        return self._trading_frac

    @property
    def effective_delta(self) -> float:
        """Alias for trading_fraction (backwards compatibility)."""
        return self._trading_frac

    def aim(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the aim portfolio.

        In "weights" mode: aim = aim_weight * target_weights.
            For a single signal, aim_weight is uniform → cancels after
            gross normalization. Only the trading fraction matters.

        In "signals" mode: aim = (γΣ)⁻¹ (aim_weight * signals).
            Per equation (15): aim_t = (γΣ)⁻¹ B (f^k / (1+φ^k a/γ))
        """
        if len(self.aim_weights) == 1:
            scaled = inputs * self.aim_weights[0]
        else:
            scaled = inputs * self.aim_weights

        if self.input_type == "signals":
            if self.cov is not None:
                return np.linalg.solve(self.gamma * self.cov, scaled)
            return scaled / self.gamma

        return scaled

    def step(
        self,
        target: np.ndarray,
        current_position: np.ndarray,
    ) -> np.ndarray:
        """
        One step of the optimal DynTrad policy.

        From equation (10):
            x_t = (1 - a/λ) · x_{t-1} + (a/λ) · aim_t

        Returns the new position, normalized to target gross exposure.
        """
        delta = self._trading_frac
        aim_t = self.aim(target)
        new_pos = (1.0 - delta) * current_position + delta * aim_t

        gross = np.abs(new_pos).sum()
        if gross > 0:
            new_pos = new_pos * (self.gross / gross)

        return new_pos

    def run(
        self,
        signal_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run DynTrad over a full signal history.

        Parameters
        ----------
        signal_df : DataFrame (dates × assets)
            Target portfolio weights (or raw signals in "signals" mode).

        Returns
        -------
        DataFrame (dates × assets) of optimally-adjusted positions.
        """
        cols = signal_df.columns
        dates = signal_df.index
        positions = {}
        w_prev = np.zeros(len(cols))

        for date in dates:
            target = signal_df.loc[date].values.astype(float)
            target = np.nan_to_num(target, 0.0)
            w_prev = self.step(target, w_prev)
            positions[date] = w_prev.copy()

        return pd.DataFrame(positions, index=cols).T

    def summary(self) -> dict:
        """Return key parameters for inspection / logging."""
        return {
            "signal_decay_phi": self.phi.tolist(),
            "risk_aversion_gamma": self.gamma,
            "cost_multiplier_lambda": self.lam,
            "discount_rate_rho": self.rho,
            "a_raw": self._a,
            "trading_fraction_delta": self._trading_frac,
            "aim_weights": self.aim_weights.tolist(),
            "gross_exposure": self.gross,
            "input_type": self.input_type,
        }


def run_dyntrad(
    raw_weights: pd.DataFrame,
    signal_decay: float | np.ndarray = 0.65,
    risk_aversion: float = 1.0,
    cost_multiplier: float = 2.0,
    discount_rate: float = 0.001,
    gross_exposure: float = 2.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Convenience wrapper: apply DynTrad to a target weight matrix.

    Parameters
    ----------
    raw_weights : DataFrame (dates × assets)
        Target portfolio weights from the signal model.
    signal_decay : float or array
        φ — signal persistence. 0.65 is the measured autocorrelation
        of the DM-RET signal in XFUND.
    risk_aversion : float
        γ — higher = more aggressive trading toward aim.
    cost_multiplier : float
        λ — the scalar in Λ = λΣ. Higher = trade slower. Default 2.0
        matches the paper's empirical application.
    discount_rate : float
        ρ — time discount per period.
    gross_exposure : float
        Target |w|₁.

    Returns
    -------
    (adjusted_weights, params_dict)
    """
    phi = np.atleast_1d(np.asarray(signal_decay, dtype=float))

    engine = DynTrad(
        signal_decay=phi,
        risk_aversion=risk_aversion,
        cost_multiplier=cost_multiplier,
        discount_rate=discount_rate,
        input_type="weights",
        gross_exposure=gross_exposure,
    )

    adjusted = engine.run(raw_weights)
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

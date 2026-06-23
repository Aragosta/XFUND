"""
BACKTEST.py

Fast, vectorized portfolio backtest from (time-varying) weights.

Assumes you obtain Yahoo price data elsewhere and pass it in as a DataFrame
(typically Adj Close), indexed by date with tickers as columns.

Core output: returns, annualized return, Sharpe, and drawdown.

FUNCTION OVERVIEW:
==================

1. **`backtest()`** - Main public function
   - Input: weights (DataFrame) + prices (DataFrame)
   - Uses drift mode: Rebalances only on signal_dates, weights drift between rebalances
   - If signal_dates is None, uses all dates as signal dates
   - Transaction costs applied only on rebalance dates
   - Output: Performance metrics (returns, equity, sharpe, drawdown, etc.)

2. **`oos_walk_forward_backtest()`** - Walk-forward backtest
   - Step 1: Generates weights using walk-forward (fits model on trailing window)
   - Step 2: Runs backtest on those weights (calls `backtest()`)
   - Combines weight generation + backtesting in one function
   - Uses drift mode with signal dates

3. **`results_backtest()`** - Unified visualization & analysis function
   - Handles BOTH single-strategy plots AND multi-strategy comparative analysis
   - Called automatically by `backtest(plot=True)` for single strategies
   - For multi-strategy: Input dict of strategies, get comprehensive analysis
   - Generates: Equity/drawdown plots, Summary DataFrame, yearly metrics, Fama-French analysis
   - Replaces hardcoded notebook analysis with modular function
   - One function for all backtest results visualization and reporting

"""

from __future__ import annotations

from typing import Callable

import inspect
import sys
import numpy as np
import pandas as pd
from scipy.stats import multivariate_t
from scipy.linalg import LinAlgError

# Optional numba import - @jit decorator becomes no-op if numba not available
try:
    from numba import jit
except ImportError:
    # Fallback no-op decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    # Fallback: define dummy Parallel and delayed if joblib not available
    def Parallel(*args, **kwargs):
        class DummyParallel:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, iterable):
                return list(iterable)
        return DummyParallel()
    def delayed(func):
        return func


def _as_prices_df(prices: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        return prices.to_frame()
    if isinstance(prices, pd.DataFrame):
        return prices
    raise TypeError("`prices` must be a pandas Series or DataFrame.")


def _as_weights_df(
    weights: pd.Series | pd.DataFrame | np.ndarray | list,
    *,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    if len(index) == 0:
        raise ValueError("`prices` has no rows.")

    if isinstance(weights, pd.DataFrame):
        w = weights.copy()
        w = w.reindex(columns=columns)
        if w.index.equals(index):
            return w.fillna(0.0)

        w = w.sort_index()
        try:
            return w.reindex(index=index, method="ffill").fillna(0.0)
        except TypeError as exc:
            if len(w.index) == len(index):
                w = w.copy()
                w.index = index
                return w.fillna(0.0)
            raise ValueError(
                "`weights` DataFrame index must align to `prices` index (same index, "
                "or a sortable subset for forward-fill)."
            ) from exc

    if isinstance(weights, pd.Series):
        if weights.index.difference(columns).empty:
            row = weights.reindex(columns).to_numpy(dtype=float, copy=False)
            w = pd.DataFrame([row], index=[index[0]], columns=columns)
            return w.reindex(index=index, method="ffill").fillna(0.0)

        if len(columns) == 1 and weights.index.difference(index).empty:
            w = weights.sort_index().to_frame(name=columns[0]).copy()
            return w.reindex(index=index, method="ffill").fillna(0.0)

        raise ValueError(
            "`weights` Series must be indexed by tickers (static weights), or by "
            "dates only when `prices` has a single column."
        )

    arr = np.asarray(weights, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != len(columns):
            raise ValueError("1D `weights` length must match number of `prices` columns.")
        w = pd.DataFrame([arr], index=[index[0]], columns=columns)
        return w.reindex(index=index, method="ffill").fillna(0.0)

    if arr.ndim == 2:
        if arr.shape[1] != len(columns):
            raise ValueError(
                "2D `weights` must have shape (T, N) with N == len(prices.columns)."
            )
        if arr.shape[0] == len(index):
            w = pd.DataFrame(arr, index=index, columns=columns)
            return w.fillna(0.0)
        if arr.shape[0] == 1:
            w = pd.DataFrame(arr, index=[index[0]], columns=columns)
            return w.reindex(index=index, method="ffill").fillna(0.0)
        raise ValueError(
            "2D `weights` must have T == len(prices.index), or T == 1 for a static vector."
        )

    raise ValueError("`weights` must be 1D or 2D when provided as an array.")


def results_backtest(
    strategies: dict[str, dict] | dict,
    *,
    baseline: dict | list[dict] | dict[str, dict] | None = None,
    title: str | None = None,
    fama_french: bool = True,
    fama_start_date: str | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> dict:
    """
    Comprehensive backtest results analysis for single or multiple strategies.

    This unified function handles all backtest visualization and analysis:
    - Equity curves (normalized to start at 1.0)
    - Drawdown curves
    - Summary DataFrame with all key metrics
    - Yearly performance metrics
    - Fama-French 3-factor analysis (optional, with significance stars)

    Can be used for both single-strategy visualization (replaces plot_backtest)
    and multi-strategy comparative analysis.

    Parameters
    ----------
    strategies : dict[str, dict] | dict
        Either:
        - Dictionary mapping strategy names to backtest result dictionaries
          (for multi-strategy analysis)
        - Single backtest result dictionary (for single-strategy visualization)
        Each result dict should contain keys: 'equity', 'returns', 'drawdown',
        and performance metrics ('ann_return', 'sharpe', etc.).

    baseline : dict | list[dict] | dict[str, dict] | None, optional
        Optional baseline(s) for comparison (used when strategies is a single dict).
        Can be:
        - Single result dict (with 'equity' key)
        - List of result dicts
        - Dict mapping names to result dicts
        - None (no baseline)

    title : str, optional
        Title for the plot. Default: None.

    fama_french : bool, default=True
        Whether to run Fama-French 3-factor regression analysis.
        Requires pandas_datareader and statsmodels packages.

    fama_start_date : str, optional
        Start date for Fama-French data (YYYY-MM-DD format).
        If None, uses the earliest date from strategy returns.

    figsize : tuple[float, float], default=(14, 8)
        Figure size for plots (width, height).

    Returns
    -------
    dict
        Dictionary containing:
        - 'fig': matplotlib Figure object with equity and drawdown plots
        - 'axes': tuple of (ax_equity, ax_drawdown) matplotlib Axes
        - 'summary_df': pandas DataFrame with overall performance metrics
        - 'yearly_df': pandas DataFrame with year-by-year metrics
        - 'fama_french_df': pandas DataFrame with factor analysis (if fama_french=True)

    Examples
    --------
    Multi-strategy analysis:
    >>> strategies = {
    ...     'Momentum': result_momentum,
    ...     'Reversion': result_reversion,
    ...     'SPY': result_spy
    ... }
    >>> results = results_backtest(strategies, title="Strategy Comparison")
    >>> display(results['summary_df'])
    >>> results['fig'].show()

    Single-strategy visualization:
    >>> results = results_backtest(
    ...     result_momentum,
    ...     baseline=result_spy,
    ...     title="Momentum Strategy"
    ... )
    >>> results['fig'].show()
    """
    import matplotlib.pyplot as plt

    # Handle single strategy input (backward compatibility with plot_backtest)
    if isinstance(strategies, dict) and 'equity' in strategies:
        # Single strategy case
        strategy_name = strategies.get('name', 'Strategy')
        strategies = {strategy_name: strategies}

        # Convert baseline to strategies dict
        if baseline is not None:
            if isinstance(baseline, dict) and 'equity' in baseline:
                # Single baseline
                baseline_name = baseline.get('name', 'Baseline')
                strategies[baseline_name] = baseline
            elif isinstance(baseline, (list, tuple)):
                # List of baselines
                for i, b in enumerate(baseline):
                    bname = b.get('name', f'Baseline{i+1}')
                    strategies[bname] = b
            elif isinstance(baseline, dict):
                # Dict of baselines
                strategies.update(baseline)

    if len(strategies) == 0:
        raise ValueError("At least one strategy must be provided")

    # ================================================================================
    # 1. EQUITY & DRAWDOWN PLOTS
    # ================================================================================

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]}
    )
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    # Collect all equity curves and align to common date range
    equity_curves = {}
    drawdown_curves = {}

    for name, result in strategies.items():
        if 'equity' in result:
            equity_curves[name] = result['equity']
        if 'drawdown' in result:
            drawdown_curves[name] = result['drawdown']

    # Align all curves to common dates and normalize equity to start at 1.0
    if equity_curves:
        # Find common date range
        all_dates = pd.Index([])
        for eq in equity_curves.values():
            all_dates = all_dates.union(eq.index)

        # Reindex and forward-fill
        aligned_equity = {}
        for name, eq in equity_curves.items():
            aligned = eq.reindex(all_dates, method='ffill')
            # Normalize to start at 1.0
            first_valid = aligned.dropna().iloc[0] if len(aligned.dropna()) > 0 else 1.0
            aligned_equity[name] = aligned / first_valid

        # Plot equity curves
        for name, eq in aligned_equity.items():
            ax1.plot(eq.index, eq.values, label=name, linewidth=1.5)

        ax1.set_ylabel("Equity (Normalized)", fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", fontsize=10)
        ax1.set_title("Cumulative Returns", fontsize=12, pad=10)

    # Plot drawdown curves
    if drawdown_curves:
        for name, dd in drawdown_curves.items():
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, label=name)
            ax2.plot(dd.index, dd.values, linewidth=1)

        ax2.set_ylabel("Drawdown", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=10)
        ax2.set_title("Drawdown", fontsize=12, pad=10)

    plt.tight_layout()

    # ================================================================================
    # 2. SUMMARY DATAFRAME
    # ================================================================================

    summary_data = {}

    # Organized as: Returns/Sharpe -> Drawdowns/Risk -> Costs/Turnover
    metric_keys = [
        # Returns & Risk-Adjusted Performance
        ('Annual Return', 'ann_return', '.2%'),
        ('Annual Volatility', 'ann_vol', '.2%'),
        ('Sharpe Ratio', 'sharpe', '.3f'),
        ('Expectancy', 'expectancy', '.4f'),
        # ('Sortino Ratio', 'sortino_ratio', '.3f'),
        ('Total Return', 'total_return', '.2%'),
        
        # Drawdowns & Risk Metrics
        ('Max Drawdown', 'max_drawdown', '.2%'),
        ('Avg Drawdown', 'avg_drawdown', '.2%'),
        ('CDaR (95%)', 'cdar', '.2%'),
        ('CVaR (95%)', 'cvar_ann', '.2%'),
        ('Downside Deviation', 'downside_deviation', '.2%'),

        # Diversification Metrics
        ('MRC Variance', 'mrc_variance', '.4f'),
        ('Herfindahl Index', 'herfindahl', '.4f'),

        # Costs & Turnover
        ('Ann. Turnover', 'ann_turnover', '.2%'),
        ('Total Cost', 'total_cost', '.2%'),


    ]

    for name, result in strategies.items():
        strategy_metrics = {}
        for display_name, key, fmt in metric_keys:
            value = result.get(key, np.nan)
            if pd.notna(value):
                strategy_metrics[display_name] = f"{value:{fmt}}"
            else:
                strategy_metrics[display_name] = "N/A"
        summary_data[name] = strategy_metrics

    summary_df = pd.DataFrame(summary_data).T

    # Validation: Verify total_return values match equity curves
    for name, result in strategies.items():
        if 'equity' in result and 'total_return' in result:
            equity = result['equity']
            total_return = result['total_return']
            equity_valid = equity.dropna()
            if len(equity_valid) > 0:
                start_equity = equity_valid.iloc[0]
                end_equity = equity_valid.iloc[-1]
                if start_equity > 0:
                    expected_total_return = (end_equity / start_equity) - 1.0
                    if abs(total_return - expected_total_return) > 1e-6:
                        import warnings
                        warnings.warn(
                            f"Total Return Validation Failed for '{name}': "
                            f"Reported total_return={total_return:.6f} ({total_return*100:.2f}%), "
                            f"but equity curve (start={start_equity:.6f}, end={end_equity:.6f}) "
                            f"suggests total_return={expected_total_return:.6f} ({expected_total_return*100:.2f}%). "
                            f"Multiplier={end_equity/start_equity:.6f}x"
                        )

    # ================================================================================
    # 3. YEARLY METRICS
    # ================================================================================

    def calculate_yearly_metrics(returns: pd.Series, drawdown: pd.Series, equity: pd.Series, freq: int = 252) -> pd.DataFrame:
        """Calculate per-year performance metrics."""
        yearly_data = []

        for year in sorted(returns.index.year.unique()):
            year_mask = returns.index.year == year
            year_returns = returns[year_mask]
            year_drawdown = drawdown[year_mask]
            year_equity = equity[year_mask]

            if len(year_returns) == 0:
                continue

            # Only filter NaN (missing data), never filter zero-return days
            year_returns_valid = year_returns.dropna()
            year_drawdown_valid = year_drawdown.loc[year_returns_valid.index] if len(year_returns_valid) > 0 else year_drawdown

            if len(year_returns_valid) == 0:
                continue

            # Calculate yearly return using equity values (more accurate)
            year_equity_valid = year_equity.dropna()
            if len(year_equity_valid) > 0:
                start_equity = year_equity_valid.iloc[0]
                end_equity = year_equity_valid.iloc[-1]
                if start_equity > 0:
                    total_return = (end_equity / start_equity) - 1.0
                else:
                    total_return = (1 + year_returns_valid).prod() - 1
            else:
                total_return = (1 + year_returns_valid).prod() - 1

            # Annualized Sharpe ratio
            if year_returns_valid.std() > 0 and len(year_returns_valid) > 1:
                sharpe = year_returns_valid.mean() / year_returns_valid.std() * np.sqrt(freq)
            else:
                sharpe = 0.0

            # Average drawdown
            dd_neg = year_drawdown_valid[year_drawdown_valid < 0]
            avg_dd = dd_neg.mean() if len(dd_neg) > 0 else 0.0

            # Max drawdown
            max_dd = year_drawdown_valid.min()

            yearly_data.append({
                'Year': year,
                'Return': total_return,
                'Sharpe': sharpe,
                'Avg DD': avg_dd,
                'Max DD': max_dd
            })

        if len(yearly_data) == 0:
            return pd.DataFrame(columns=['Year', 'Return', 'Sharpe', 'Avg DD', 'Max DD'])

        return pd.DataFrame(yearly_data).set_index('Year')

    # Calculate yearly metrics for each strategy
    yearly_metrics_list = {}
    for name, result in strategies.items():
        if 'returns' in result and 'drawdown' in result and 'equity' in result:
            yearly = calculate_yearly_metrics(
                result['returns'],
                result['drawdown'],
                result['equity'],
                freq=252
            )
            if len(yearly) > 0:
                yearly_metrics_list[name] = yearly

    yearly_df = pd.DataFrame()

    if yearly_metrics_list:
        # Get all unique years
        all_years = sorted(set().union(*[df.index for df in yearly_metrics_list.values()]))
        yearly_df = pd.DataFrame(index=all_years)

        # Add columns for each strategy and metric
        for name, yearly in yearly_metrics_list.items():
            if 'Return' in yearly.columns:
                yearly_df[f'{name} Return'] = yearly['Return']
            if 'Sharpe' in yearly.columns:
                yearly_df[f'{name} Sharpe'] = yearly['Sharpe']
            if 'Max DD' in yearly.columns:
                yearly_df[f'{name} Max DD'] = yearly['Max DD']
            if 'Avg DD' in yearly.columns:
                yearly_df[f'{name} Avg DD'] = yearly['Avg DD']

    # ================================================================================
    # 4. FAMA-FRENCH 3-FACTOR ANALYSIS
    # ================================================================================

    fama_french_df = pd.DataFrame()

    if fama_french:
        try:
            import pandas_datareader as pdr
            from statsmodels.api import OLS, add_constant
            import warnings
            warnings.filterwarnings('ignore', category=FutureWarning, message='.*date_parser.*')

            # Determine start date
            if fama_start_date is None:
                # Use earliest date from all strategies
                min_date = None
                for result in strategies.values():
                    if 'returns' in result:
                        returns = result['returns'].dropna()
                        if len(returns) > 0:
                            date = returns.index[0]
                            if min_date is None or date < min_date:
                                min_date = date
                fama_start_date = min_date.strftime('%Y-%m-%d') if min_date else '2000-01-01'

            # Fetch Fama-French factors
            ff_factors = pdr.get_data_famafrench('F-F_Research_Data_Factors', start=fama_start_date)[0] / 100

            # Convert PeriodIndex to DatetimeIndex (month-end)
            if isinstance(ff_factors.index, pd.PeriodIndex):
                ff_factors.index = ff_factors.index.to_timestamp('M')

            def get_significance_star(p_value):
                """Return significance star: * p<0.05, ** p<0.01, *** p<0.001."""
                if p_value < 0.001:
                    return '***'
                elif p_value < 0.01:
                    return '**'
                elif p_value < 0.05:
                    return '*'
                return ''

            def run_ff_regression(strategy_returns, ff_data, strategy_name="Strategy"):
                """Run Fama-French 3-factor regression with significance stars."""

                # Drop NaN and convert to monthly
                strategy_returns_clean = strategy_returns.dropna()

                if len(strategy_returns_clean) == 0:
                    return None

                # Compound daily returns to monthly
                if not isinstance(strategy_returns_clean.index, pd.DatetimeIndex):
                    strategy_returns_clean.index = pd.to_datetime(strategy_returns_clean.index)

                monthly_returns = (1 + strategy_returns_clean).resample('ME').prod() - 1

                # Ensure month-end alignment
                if not isinstance(monthly_returns.index, pd.DatetimeIndex):
                    monthly_returns.index = pd.to_datetime(monthly_returns.index)
                    monthly_returns.index = monthly_returns.index.to_period('M').to_timestamp('M')

                # Align with Fama-French data
                ff_index = ff_data.index
                if isinstance(ff_index, pd.PeriodIndex):
                    ff_index = ff_index.to_timestamp('M')

                common_dates = monthly_returns.index.intersection(ff_index)

                if len(common_dates) < 12:
                    return None

                # Align data
                y = monthly_returns.loc[common_dates].values
                X = ff_data.loc[common_dates, ['Mkt-RF', 'SMB', 'HML']].values
                X_with_const = add_constant(X)

                # Run OLS regression
                model = OLS(y, X_with_const).fit()

                # Extract results
                alpha_monthly = model.params[0]
                alpha = alpha_monthly * 12  # Annualize
                alpha_tstat = model.tvalues[0]
                alpha_pvalue = model.pvalues[0]

                beta_mkt = model.params[1]
                beta_smb = model.params[2]
                beta_hml = model.params[3]
                adj_r2 = model.rsquared_adj

                # Add significance stars
                alpha_star = get_significance_star(alpha_pvalue)

                return {
                    'Annual Alpha': f"{alpha:.4f}{alpha_star}",
                    'Alpha t-stat': f"{alpha_tstat:.4f}{alpha_star}",
                    'Alpha p-value': f"{alpha_pvalue:.4f}{alpha_star}",
                    'β Market': f"{beta_mkt:.4f}",
                    'β Size (SMB)': f"{beta_smb:.4f}",
                    'β Value (HML)': f"{beta_hml:.4f}",
                    'Adj R²': f"{adj_r2:.4f}"
                }

            # Run regression for each strategy
            ff_results = {}
            for name, result in strategies.items():
                if 'returns' in result:
                    ff_result = run_ff_regression(result['returns'], ff_factors, name)
                    if ff_result:
                        ff_results[name] = ff_result

            if ff_results:
                fama_french_df = pd.DataFrame(ff_results).T

        except ImportError as e:
            print(f"Warning: Fama-French analysis requires pandas_datareader and statsmodels: {e}")
        except Exception as e:
            print(f"Warning: Fama-French analysis failed: {e}")

    # ================================================================================
    # RETURN RESULTS
    # ================================================================================

    return {
        'fig': fig,
        'axes': (ax1, ax2),
        'summary_df': summary_df,
        'yearly_df': yearly_df,
        'fama_french_df': fama_french_df,
    }


@jit(nopython=True, cache=True)
def _reconstruct_drifted_weights(
    rets_np: np.ndarray,
    w_target_np: np.ndarray,
    exec_indices: np.ndarray,
    n_dates: int,
    n_assets: int,
) -> np.ndarray:
    """
    Weight reconstruction for risk calculations.
    Reconstructs actual drifted weights at each period.

    Uses numba JIT compilation if available (fast), otherwise runs as pure Python (slower).
    The @jit decorator handles missing numba gracefully.
    """
    w_actual_np = np.zeros((n_dates, n_assets))
    w_current = np.zeros(n_assets)
    
    for i in range(n_dates):
        # At start of period i, check if rebalance
        if exec_indices[i] == 1:
            # Rebalance at start of period i: set to target weights
            for j in range(n_assets):
                w_current[j] = w_target_np[i, j]
        
        # Store weights at start of period i (for risk calculations)
        for j in range(n_assets):
            w_actual_np[i, j] = w_current[j]
        
        # At end of period i, update weights for period i+1 based on returns during period i
        if i > 0:
            # Calculate portfolio return
            port_ret_i = 0.0
            for j in range(n_assets):
                ret_val = rets_np[i, j]
                if np.isfinite(ret_val):
                    port_ret_i += w_current[j] * ret_val
            
            if abs(port_ret_i) > 1e-10:
                # Drift formula: w_new[j] = (w_old[j] * (1 + r[j])) / (1 + rp)
                denom = 1.0 + port_ret_i
                for j in range(n_assets):
                    ret_val = rets_np[i, j]
                    if np.isfinite(ret_val):
                        w_current[j] = (w_current[j] * (1.0 + ret_val)) / denom
                        if not np.isfinite(w_current[j]):
                            w_current[j] = 0.0
                    # If ret_val is not finite, weight stays the same (already set above)
    
    return w_actual_np


def _get_oos_period_boundaries(
    prev_refit_date: pd.Timestamp,
    current_refit_date: pd.Timestamp,
    returns_index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """
    Calculate OOS period boundaries for a given refit period.
    
    OOS period is defined as: (prev_refit_date, current_refit_date]
    This is used both in the callback and in hyperparameter optimization to ensure consistency.
    
    Parameters:
    -----------
    prev_refit_date : pd.Timestamp
        Start of OOS period (exclusive)
    current_refit_date : pd.Timestamp
        End of OOS period (inclusive)
    returns_index : pd.DatetimeIndex
        Index of returns series
        
    Returns:
    --------
    pd.DatetimeIndex: OOS period dates
    """
    return returns_index[(returns_index > prev_refit_date) & (returns_index <= current_refit_date)]


def backtest(
    weights: pd.Series | pd.DataFrame | np.ndarray | list,
    prices: pd.Series | pd.DataFrame,
    *,
    freq: int = 252,
    lag: int = 1,
    risk_free_rate: float = 0.0,
    signal_dates: list | None = None,
    transaction_cost: float = 0.0,
    compute_risk_metrics: bool = True,
    baseline: str | None = None,
) -> dict:
    """
    Vectorized backtest for a portfolio defined by weights.

    Uses drift mode: Rebalances only on signal_dates, weights drift between rebalances,
    and transaction costs are applied. If signal_dates is None, all dates are used as signal dates.

    Args:
        weights:
            - pd.DataFrame: index=dates, columns=tickers (time-varying weights)
            - pd.Series: index=tickers (static weights), or index=dates for single-asset `prices`
            - 1D array-like: length N (static weights, aligned to prices.columns order)
            - 2D array-like: shape (T, N) aligned by row order to `prices.index`
        prices:
            pd.DataFrame (or Series) of prices (e.g. Yahoo Adj Close), indexed by date.
        freq: Annualization factor (252 for daily).
        lag:
            Apply weights with this lag to avoid look-ahead (default 1). With daily close data,
            `returns[t]` is the return from `t-1 -> t`.
            
            TIMING DETAILS:
            - Signal generated at time t (signal_pos)
            - Execution occurs at time t + lag + 1 (exec_pos = signal_pos + lag + 1)
            - Weights from signal time t are applied at START of period exec_pos
            - Return for period exec_pos (from exec_pos-1 to exec_pos) uses these weights
            - This ensures no same-day signal → same-day execution (look-ahead bias)
            - With lag=1: signal at t → execution at t+2 → weights applied to return from t+1 to t+2
            - With lag=2: signal at t → execution at t+3 → weights applied to return from t+2 to t+3
        risk_free_rate: Annual risk-free rate for Sharpe (default 0.0).
        signal_dates:
            List of dates when rebalancing signals are generated. If None, all dates are used.
        transaction_cost:
            Transaction cost per unit traded notional (applied when weights change).
            Turnover per date is computed as `0.5 * sum(abs(delta_w))` on executed weights (after `lag`).
        compute_risk_metrics:
            If True, compute MRC variance and ENB (risk contribution metrics).
            Set to False to skip risk calculations for ~20-30% speedup when not needed.
        baseline:
            Ticker name (e.g., "SPY") for buy-and-hold baseline comparison. 
            Calculated with no transaction costs. If None, no baseline is computed.

    Returns:
        dict with:
            returns (pd.Series), ann_return (float), sharpe (float),
            ic (float), drawdown (pd.Series), max_drawdown (float)
    """
    if lag < 0:
        raise ValueError("`lag` must be >= 0.")
    if transaction_cost < 0:
        raise ValueError("`transaction_cost` must be >= 0.")

    px = _as_prices_df(prices).sort_index()
    if px.shape[1] == 0:
        raise ValueError("`prices` has no columns.")
    if px.shape[0] == 0:
        raise ValueError("`prices` has no rows.")

    w_target = _as_weights_df(weights, index=px.index, columns=px.columns)

    # Backtest requires signal_dates for drift mode
    if signal_dates is None:
        # Use all dates as signal dates if not provided
        signal_dates = px.index.tolist()

    idx = px.index
    n_dates = len(idx)
    n_assets = len(px.columns)

    rets = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    # Build execution indicator array and w_target array - vectorized
    exec_indices = np.zeros(n_dates, dtype=np.int32)
    w_target_np = np.zeros((n_dates, n_assets), dtype=np.float64)

    # Convert signal dates to timestamps once
    signal_ts = [pd.Timestamp(x) for x in signal_dates]

    # Optimization: Convert weights to numpy for fast indexing inside loop
    w_target_vals = w_target.values

    # Vectorized processing of signal dates
    # TIMING: Signal at time t → Execution at time t + lag + 1
    # This ensures weights from signal time t are applied to returns starting at t + lag + 1
    # With lag=1: signal at t → execution at t+2 → weights applied to return from t+1 to t+2
    # This prevents same-day signal → same-day execution (look-ahead bias)
    for t in signal_ts:
        if t not in idx:
            continue
        signal_pos = idx.get_loc(t)
        exec_pos = signal_pos + lag + 1  # Execution happens lag+1 periods after signal
        if exec_pos >= n_dates:
            continue
        exec_indices[exec_pos] = 1  # Mark execution at exec_pos

        # Copy weights from signal date to execution date - vectorized
        # Weights from signal time t are stored at execution position exec_pos
        # Use direct numpy indexing (signal_pos) instead of slow DataFrame.loc[t]
        w_signal = w_target_vals[signal_pos]
        w_target_np[exec_pos, :] = w_signal

        # Handle missing prices at execution - vectorized
        missing_px_vals = px.iloc[exec_pos].isna().values
        if missing_px_vals.any():
            w_target_np[exec_pos, missing_px_vals] = 0.0

    rets_np = rets.to_numpy(dtype=np.float64, copy=True)

    # Call compiled core function
    equity_np, _, turnover_gross_np, cost_frac_np, portfolio_returns_np = _backtest_drift_tc_core(
        rets_np, w_target_np, exec_indices, transaction_cost, n_assets, n_dates
    )

    equity = pd.Series(equity_np, index=idx)
    turnover_gross = pd.Series(turnover_gross_np, index=idx, name="turnover_gross")
    turnover = (0.5 * turnover_gross).rename("turnover")
    cost_frac = pd.Series(cost_frac_np, index=idx, name="cost_frac")

    # VALIDATION: Ensure equity starts at 1.0 (critical for annual return calculation)
    if len(equity) > 0 and abs(equity.iloc[0] - 1.0) > 1e-6:
        import warnings
        warnings.warn(
            f"Equity normalization warning: Equity does not start at 1.0. "
            f"equity.iloc[0]={equity.iloc[0]:.8f}. "
            f"Date range: {equity.index[0]} to {equity.index[-1]}, "
            f"n_dates={len(equity)}. This may cause incorrect annual return calculation."
        )
        # Normalize equity to start at 1.0 if it doesn't
        if equity.iloc[0] > 0:
            equity = equity / equity.iloc[0]

    # FIX: Calculate net returns directly from equity to ensure consistency
    # This automatically includes all transaction costs (including deferred costs from period 0)
    # and avoids the double-counting issue where equity and returns diverge
    net_ret = equity.pct_change(fill_method=None).fillna(0.0)
    net_ret.iloc[0] = 0.0  # First period always has no return
    net_ret = net_ret.rename("returns")

    drawdown = equity / equity.cummax() - 1.0
    
    # Calculate additional drawdown metrics
    # Percent of Time in Drawdown: Fraction of total time the strategy is below its previous peak
    valid_dd = drawdown.dropna()
    percent_time_in_drawdown = float((valid_dd < 0).sum() / len(valid_dd)) if len(valid_dd) > 0 else np.nan

    # Conditional Drawdown at Risk (CDaR): Expected drawdown beyond a percentile (default 95th)
    # Similar to CVaR but for drawdowns
    cdar_percentile = 0.95
    dd_vals = drawdown.dropna().values
    if len(dd_vals) > 0:
        dd_sorted = np.sort(dd_vals)  # Sort ascending (most negative first)
        threshold_idx = int(np.ceil((1 - cdar_percentile) * len(dd_sorted)))
        if threshold_idx < len(dd_sorted):
            tail_drawdowns = dd_sorted[:threshold_idx]
            cdar = float(np.mean(tail_drawdowns)) if len(tail_drawdowns) > 0 else np.nan
        else:
            cdar = float(np.mean(dd_sorted)) if len(dd_sorted) > 0 else np.nan
    else:
        cdar = np.nan

    # Calculate Variance of Marginal Risk Contributions (MRC) and Herfindahl Index
    # These metrics require portfolio weights and covariance matrix
    # Only compute if requested (can be expensive)
    if compute_risk_metrics:
        cov_window = 60
        mrc_variance_series = pd.Series(np.nan, index=idx, name="mrc_variance")
        herfindahl_series = pd.Series(np.nan, index=idx, name="herfindahl")
        
        # Get asset returns for covariance calculation
        asset_ret = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        
        # Reconstruct actual drifted weights at each period
        # Uses numba JIT if available (fast), otherwise pure Python (slower)
        rets_np = asset_ret.to_numpy(dtype=np.float64, copy=False)
        w_actual_np = _reconstruct_drifted_weights(
            rets_np, w_target_np, exec_indices, n_dates, n_assets
        )
        
        # Convert to DataFrame for risk calculations
        w_actual = pd.DataFrame(w_actual_np, index=idx, columns=px.columns)
        
        # Calculate rolling covariance and risk contributions
        # Vectorize where possible
        for i in range(cov_window + 1, len(idx)):
            # Get returns window for covariance estimation
            # FIX: Include returns up to i-1 (inclusive) since at time i we know return at i-1
            # At time i, we calculate risk using weights from start of period i
            # These weights are based on information up to close of period i-1
            # So covariance should include all returns up to i-1
            # Window: [i - cov_window:i] gives us cov_window periods ending at i-1
            window_start = max(0, i - cov_window)
            window_end = i  # Python slicing is exclusive, so this gives us up to i-1 inclusive
            ret_window = asset_ret.iloc[window_start:window_end]
            w_curr = w_actual.iloc[i]
            
            # Skip if no valid data
            if ret_window.isna().all().all() or w_curr.abs().sum() == 0:
                continue
            
            # Calculate covariance matrix (annualized)
            cov_matrix = ret_window.cov() * freq
            
            # Get valid assets (non-zero weights and valid covariance)
            valid_assets = w_curr[w_curr.abs() > 1e-8].index
            if len(valid_assets) < 2:
                continue
            
            # Filter covariance to valid assets
            cov_subset = cov_matrix.loc[valid_assets, valid_assets]
            w_subset = w_curr[valid_assets].values
            
            # Portfolio volatility: sqrt(w^T * Σ * w)
            port_var = w_subset @ cov_subset @ w_subset
            if port_var <= 0 or not np.isfinite(port_var):
                continue
            port_vol = np.sqrt(port_var)
            
            # Marginal Risk Contribution: MRC_i = (Σw)_i / σ_p
            sigma_w = cov_subset @ w_subset
            mrc = sigma_w / port_vol if port_vol > 1e-8 else np.zeros_like(sigma_w)
            ctr = w_subset * mrc
            
            # Variance of MRC across assets
            if len(mrc) > 1:
                mrc_variance = float(np.var(mrc))
                mrc_variance_series.iloc[i] = mrc_variance

            # Herfindahl Index: sum of squared weights (concentration measure)
            # Range: [1/N, 1] where lower = more diversified, higher = more concentrated
            herfindahl = float(np.sum(w_subset ** 2))
            herfindahl_series.iloc[i] = herfindahl
        
        # Average metrics over the backtest period
        mrc_variance_avg = float(np.nanmean(mrc_variance_series.values)) if np.isfinite(mrc_variance_series.values).any() else np.nan
        herfindahl_avg = float(np.nanmean(herfindahl_series.values)) if np.isfinite(herfindahl_series.values).any() else np.nan
    else:
        # Skip risk calculations
        mrc_variance_series = pd.Series(np.nan, index=idx, name="mrc_variance")
        herfindahl_series = pd.Series(np.nan, index=idx, name="herfindahl")
        mrc_variance_avg = np.nan
        herfindahl_avg = np.nan
    
    # CRITICAL: Only filter NaN (missing data), never filter zero-return days
    # Zero returns are valid and must be included in Sharpe calculation
    # Filtering zeros would artificially inflate Sharpe ratio
    ret_vals = net_ret[net_ret.notna()].values
    n_obs = len(ret_vals)

    # Annualize return using discrete compounding with trading periods
    # This is consistent with how equity is built (discrete compounding)
    if n_obs > 0:
        total_return_factor = equity.iloc[-1]  # Starts at 1.0, so this is the total return factor
        periods_per_year = float(freq)
        n_trading_days = len(equity)  # Calculate trading days upfront

        # DEBUG: Log annual return calculation parameters for diagnosis
        # Enable via environment variable: BACKTEST_DEBUG_ANNRET=1
        import os
        if os.getenv('BACKTEST_DEBUG_ANNRET', '0') == '1':
            start_equity_val = equity.iloc[0] if len(equity) > 0 else np.nan
            end_equity_val = equity.iloc[-1] if len(equity) > 0 else np.nan
            date_range_str = f"{equity.index[0]} to {equity.index[-1]}" if len(equity) > 0 else "N/A"
            import warnings
            warnings.warn(
                f"[BACKTEST_DEBUG] Annual Return Calculation:\n"
                f"  Date range: {date_range_str}\n"
                f"  Trading days (equity length): {n_trading_days}\n"
                f"  Valid return observations (n_obs): {n_obs}\n"
                f"  Equity start: {start_equity_val:.8f} (expected: 1.0)\n"
                f"  Equity end: {end_equity_val:.8f}\n"
                f"  Total return factor: {total_return_factor:.8f}\n"
                f"  Periods per year (freq): {periods_per_year}\n"
                f"  Annualization exponent: {periods_per_year / n_trading_days:.6f}\n"
                f"  Ann return formula: ({total_return_factor} ** ({periods_per_year} / {n_trading_days})) - 1"
            )

        # PRIMARY METHOD: Annualize using actual trading days (equity length)
        # This is the correct method as it uses actual calendar time, not filtered observations
        # Using n_obs would artificially inflate returns when there are many NaN days
        ann_return = float((total_return_factor ** (periods_per_year / n_trading_days)) - 1.0)

        # ALTERNATIVE METHOD: Annualize using valid return observations (for comparison)
        # This method can overstate returns if many NaN returns exist
        # Kept for backward compatibility and debugging purposes
        if n_trading_days != n_obs:
            ann_return_alt = float((total_return_factor ** (periods_per_year / n_obs)) - 1.0)

            # Log discrepancy if significant (debug mode only)
            import os
            if os.getenv('BACKTEST_DEBUG_ANNRET', '0') == '1':
                diff_pct = abs(ann_return - ann_return_alt) / (abs(ann_return) + 1e-10) * 100
                if diff_pct > 1.0:  # More than 1% difference
                    import warnings
                    warnings.warn(
                        f"[BACKTEST_DEBUG] Annual return method discrepancy:\n"
                        f"  Method 1 (n_trading_days={n_trading_days}): {ann_return:.4%}\n"
                        f"  Method 2 (n_obs={n_obs}): {ann_return_alt:.4%}\n"
                        f"  Difference: {diff_pct:.2f}%\n"
                        f"  n_obs vs n_trading_days difference: {n_trading_days - n_obs} days"
                    )
        else:
            ann_return_alt = ann_return  # Methods agree when n_obs == n_trading_days
    else:
        ann_return = np.nan
    ann_vol = float(np.std(ret_vals, ddof=1) * np.sqrt(freq)) if n_obs > 1 else np.nan
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / freq) - 1.0
    sharpe = np.nan
    if n_obs > 1 and ann_vol > 0:
        ret_mean = float(np.mean(ret_vals))
        sharpe = float(((ret_mean - rf_daily) / np.std(ret_vals, ddof=1)) * np.sqrt(freq))

    # Conditional Value at Risk (CVaR 95%): Expected Shortfall - average of worst 5% returns
    cvar_percentile = 0.95
    ret_vals_cvar = net_ret.dropna().values
    if len(ret_vals_cvar) > 0:
        ret_sorted = np.sort(ret_vals_cvar)  # Sort ascending (most negative first)
        threshold_idx = int(np.ceil((1 - cvar_percentile) * len(ret_sorted)))
        if threshold_idx > 0 and threshold_idx < len(ret_sorted):
            tail_returns = ret_sorted[:threshold_idx]
            cvar = float(np.mean(tail_returns)) if len(tail_returns) > 0 else np.nan
        else:
            cvar = float(np.mean(ret_sorted)) if len(ret_sorted) > 0 else np.nan
        # Annualize CVaR
        cvar_ann = cvar * np.sqrt(freq) if np.isfinite(cvar) else np.nan
    else:
        cvar = np.nan
        cvar_ann = np.nan

    # Downside Deviation: Standard deviation of negative returns only (for Sortino ratio)
    negative_returns = net_ret[net_ret < 0].dropna()
    if len(negative_returns) > 1:
        downside_deviation = float(negative_returns.std(ddof=1) * np.sqrt(freq))
    else:
        downside_deviation = np.nan

    # Sortino Ratio: (Return - RFR) / Downside Deviation
    if np.isfinite(downside_deviation) and downside_deviation > 0 and np.isfinite(ann_return):
        sortino_ratio = float((ann_return - risk_free_rate) / downside_deviation)
    else:
        sortino_ratio = np.nan

    cost_amount = pd.Series(0.0, index=idx, name="cost_amount")
    denom = 1.0 - cost_frac.fillna(0.0)
    mask = cost_frac.fillna(0.0) > 0
    safe = mask & (denom > 1e-12)
    cost_amount.loc[safe] = equity.loc[safe] * (cost_frac.loc[safe] / denom.loc[safe])

    turnover_vals = turnover.to_numpy(dtype=float, copy=False)
    cost_vals = cost_amount.to_numpy(dtype=float, copy=False)
    avg_turnover = float(np.nanmean(turnover_vals)) if np.isfinite(turnover_vals).any() else np.nan
    ann_turnover = avg_turnover * float(freq) if np.isfinite(avg_turnover) else np.nan
    total_turnover = float(np.nansum(turnover_vals)) if np.isfinite(turnover_vals).any() else np.nan
    total_cost = float(np.nansum(cost_vals)) if np.isfinite(cost_vals).any() else np.nan

    # Calculate average drawdown (mean of all negative drawdown values)
    dd_negative = drawdown[drawdown < 0]
    avg_drawdown = float(dd_negative.mean()) if len(dd_negative) > 0 else 0.0

    # Calculate expectancy: (Win Rate × Average Win) - (Loss Rate × Average Loss)
    # Filter out NaN returns for expectancy calculation
    valid_returns = net_ret.dropna()
    if len(valid_returns) > 0:
        positive_returns = valid_returns[valid_returns > 0]
        negative_returns = valid_returns[valid_returns < 0]
        zero_returns = valid_returns[valid_returns == 0]
        
        n_total = len(valid_returns)
        n_wins = len(positive_returns)
        n_losses = len(negative_returns)
        
        win_rate = n_wins / n_total if n_total > 0 else 0.0
        loss_rate = n_losses / n_total if n_total > 0 else 0.0
        
        avg_win = float(positive_returns.mean()) if n_wins > 0 else 0.0
        avg_loss = float(negative_returns.mean()) if n_losses > 0 else 0.0  # Already negative, keep as is
        
        # Expectancy = (Win Rate × Average Win) - (Loss Rate × Average Loss)
        # Note: avg_loss is already negative, so we subtract it (which adds the absolute value)
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    else:
        win_rate = 0.0
        loss_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        expectancy = 0.0

    # Calculate total return using first and last valid equity values (more robust)
    # Formula: total_return = (end_equity / start_equity) - 1.0
    equity_valid = equity.dropna()
    if len(equity_valid) > 0:
        start_equity = equity_valid.iloc[0]
        end_equity = equity_valid.iloc[-1]
        if start_equity > 0:
            multiplier = end_equity / start_equity
            total_return = float(multiplier - 1.0)
            
            # Validation: Check if equity starts at 1.0 (expected behavior)
            if abs(start_equity - 1.0) > 1e-6:
                import warnings
                warnings.warn(
                    f"Total Return Warning: Equity does not start at 1.0. "
                    f"start_equity={start_equity:.6f}, end_equity={end_equity:.6f}, "
                    f"multiplier={multiplier:.6f}x, total_return={total_return:.6f} ({total_return*100:.2f}%)"
                )
            
            # Validation: Verify calculation consistency
            # Recalculate to ensure formula is correct: (end/start) - 1
            recalculated = (end_equity / start_equity) - 1.0
            if abs(total_return - recalculated) > 1e-10:
                import warnings
                warnings.warn(
                    f"Total Return Calculation Error: Mismatch detected. "
                    f"total_return={total_return:.6f}, recalculated={recalculated:.6f}"
                )
        else:
            # Fallback: assume equity started at 1.0 if start_equity <= 0
            total_return = float(equity.iloc[-1] - 1.0) if len(equity) > 0 else 0.0
    else:
        # Fallback: assume equity started at 1.0 if no valid equity values
        total_return = float(equity.iloc[-1] - 1.0) if len(equity) > 0 else 0.0

    result = {
        "returns": net_ret,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "avg_drawdown": avg_drawdown,
        "equity": equity,
        "drawdown": drawdown,
        "turnover_gross": turnover_gross,
        "turnover": turnover,
        "cost_frac": cost_frac,
        "cost_amount": cost_amount,
        "avg_turnover": avg_turnover,
        "ann_turnover": ann_turnover,
        "total_turnover": total_turnover,
        "total_cost": total_cost,
        "transaction_cost": float(transaction_cost),
        # Additional drawdown metrics
        "cdar": cdar,
        "percent_time_in_drawdown": percent_time_in_drawdown,
        # Tail risk metrics
        "cvar": cvar,
        "cvar_ann": cvar_ann,
        # Downside risk metrics
        "downside_deviation": downside_deviation,
        "sortino_ratio": sortino_ratio,
        # Risk contribution metrics
        "mrc_variance": mrc_variance_avg,
        "mrc_variance_series": mrc_variance_series,
        # Diversification metrics
        "herfindahl": herfindahl_avg,
        "herfindahl_series": herfindahl_series,
        # Expectancy metrics
        "expectancy": float(expectancy),
        "win_rate": float(win_rate),
        "loss_rate": float(loss_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
    }

    # Calculate baseline if provided
    if baseline is not None:
        if baseline not in px.columns:
            raise ValueError(f"Baseline ticker `{baseline}` not found in prices columns.")
        baseline_px = px[[baseline]].copy()
        baseline_weights = pd.DataFrame(1.0, index=baseline_px.index, columns=baseline_px.columns)
        # For buy-and-hold: rebalance once at the start with lag=0 (immediate execution)
        # This ensures weights are set from the very first period and no further rebalancing
        baseline_result = backtest(
            baseline_weights,
            baseline_px,
            signal_dates=[baseline_px.index[0]],
            freq=freq,
            lag=0,  # lag=0 for immediate execution from first period
            risk_free_rate=risk_free_rate,
            transaction_cost=0.0,  # No transaction costs for baseline
            baseline=None,  # Don't calculate baseline of baseline
        )
        baseline_result["name"] = baseline
        result["baseline"] = baseline_result

    return result


def _rebalance_dates(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    """
    Return the last *observed* timestamp in each resample bin.

    This differs from using `resample(rule).last().index`, which yields the bin labels
    (e.g. calendar month-end) that may not exist in a trading-day index.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("`index` must be a pandas DatetimeIndex.")
    if len(index) == 0:
        return pd.DatetimeIndex([])

    # Normalize deprecated offset aliases to new format
    # 'M' (month end) is deprecated, use 'ME' instead
    if rule == 'M':
        rule_normalized = 'ME'
    elif rule.startswith('M-'):
        # Handle cases like 'M-MON' -> 'ME-MON'
        rule_normalized = 'ME' + rule[1:]
    else:
        rule_normalized = rule

    idx = index.sort_values()
    last_in_bin = idx.to_series().resample(rule_normalized).last().dropna()
    dates = pd.DatetimeIndex(last_in_bin.to_numpy())
    result = dates.intersection(idx).unique()
    return result

@jit(nopython=True, cache=True)
def _backtest_drift_tc_core(
    rets_np: np.ndarray,
    w_target_np: np.ndarray,
    exec_indices: np.ndarray,
    transaction_cost: float,
    n_assets: int,
    n_dates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-JIT compiled core loop for backtest_drift_tc.

    Returns: (equity, final_weights, turnover_gross, cost_frac, portfolio_returns)
    """
    equity = np.ones(n_dates)
    w = np.zeros(n_assets)
    turnover_gross = np.zeros(n_dates)
    cost_frac = np.zeros(n_dates)
    portfolio_returns = np.zeros(n_dates)

    for i in range(n_dates):
        # TIMING: Rebalance happens at START of period i (before calculating returns for period i)
        # If exec_indices[i] == 1, weights are updated at start of period i
        # Then return for period i (from i-1 to i) is calculated using these updated weights
        # This ensures proper lag application: signal at t → execution at t+lag+1 → return from t+lag to t+lag+1
        if exec_indices[i] == 1:
            # Calculate turnover (before updating weights)
            t_over = 0.0
            for j in range(n_assets):
                w_new_j = w_target_np[i, j]
                t_over += abs(w_new_j - w[j])

            # Update weights to target (rebalance happens at start of period i)
            for j in range(n_assets):
                w[j] = w_target_np[i, j]

            # Apply transaction cost
            # Use one-way turnover for a self-financing rebalance.
            # Only apply costs after we've earned at least one return (i > 0)
            # For period 0, costs are deferred to period 1 to avoid reducing starting capital
            if i > 0:
                tc_drag = transaction_cost * (0.5 * t_over)
                if tc_drag > 0.999:
                    tc_drag = 0.999
                turnover_gross[i] = t_over
                cost_frac[i] = tc_drag
                # Cost will be applied after calculating return
            else:
                # Period 0: record turnover but defer cost to period 1
                turnover_gross[i] = t_over
                cost_frac[i] = 0.0

        # Calculate portfolio return for period i (using current weights after rebalance)
        if i > 0:
            # Calculate portfolio return using current weights
            rp = 0.0
            for j in range(n_assets):
                r_val = rets_np[i, j]
                if np.isnan(r_val):
                    r_val = 0.0
                rp += w[j] * r_val

            # Store actual portfolio return (before transaction costs)
            portfolio_returns[i] = rp

            equity[i] = equity[i - 1] * (1.0 + rp)

            # Apply transaction cost if rebalance occurred at period i
            if exec_indices[i] == 1 and i > 0:
                equity[i] = equity[i] * (1.0 - cost_frac[i])

            # Update weights due to drift (for next period)
            denom = 1.0 + rp
            if np.isfinite(denom) and denom > 0:
                for j in range(n_assets):
                    r_val = rets_np[i, j]
                    if np.isnan(r_val):
                        r_val = 0.0
                    w[j] = (w[j] * (1.0 + r_val)) / denom
        
        # Apply deferred transaction cost from period 0 rebalance
        if i == 1 and exec_indices[0] == 1:
            # Apply the cost that was deferred from period 0
            t_over_deferred = turnover_gross[0]
            tc_drag = transaction_cost * (0.5 * t_over_deferred)
            if tc_drag > 0.999:
                tc_drag = 0.999
            cost_frac[0] = tc_drag
            # Apply cost to equity at period 1 (after first return)
            equity[1] = equity[1] * (1.0 - tc_drag)
    
    # Period 0 has no return (starting period)
    portfolio_returns[0] = 0.0

    return equity, w, turnover_gross, cost_frac, portfolio_returns


def position_engine(
    prev_positions: set[str] | pd.Series,
    signals: dict,
    *,
    entry_threshold: float | None = None,
    exit_threshold: float | None = None,
    entry_quantile: float | None = None,
    exit_quantile: float | None = None,
    entry_quantile_long: float | None = None,
    exit_quantile_long: float | None = None,
    entry_quantile_short: float | None = None,
    exit_quantile_short: float | None = None,
    min_hold_periods: int = 0,
    cooldown_periods: int = 0,
) -> set[str]:
    """
    Update positions based on entry/exit signals.
    
    Manages position state by applying entry/exit logic based on signals.
    This separates position management from alpha generation and sizing.
    Supports both long and short positions with separate quantile thresholds.
    
    Args:
        prev_positions: Previous set of active positions (asset names)
        signals: Dict with 'alpha', 'entry', 'exit' keys (all pd.Series indexed by assets)
        entry_threshold: Absolute threshold for entry (if entry signal is numeric)
        exit_threshold: Absolute threshold for exit (if exit signal is numeric)
        entry_quantile: Quantile threshold for entry (uses alpha if entry not provided)
                       Backward compatibility: if provided, used for long positions only
        exit_quantile: Quantile threshold for exit (uses alpha if exit not provided)
                      Backward compatibility: if provided, used for long positions only
        entry_quantile_long: Quantile threshold for long entry (e.g., 0.9 = 90th percentile)
                            Enter long when alpha >= this quantile
        exit_quantile_long: Quantile threshold for long exit (e.g., 0.9 = 90th percentile)
                           Exit long when alpha < this quantile
        entry_quantile_short: Quantile threshold for short entry (e.g., 0.1 = 10th percentile)
                             Enter short when alpha <= this quantile
        exit_quantile_short: Quantile threshold for short exit (e.g., 0.1 = 10th percentile)
                            Exit short when alpha > this quantile
        min_hold_periods: Minimum periods to hold position (prevents immediate re-entry)
        cooldown_periods: Periods to wait after exit before re-entry (not yet implemented)
    
    Returns:
        set[str]: New set of active positions (both long and short; sizing engine determines sign)
    """
    if isinstance(prev_positions, pd.Series):
        # Handle both positive and negative weights (long and short positions)
        prev_positions = set(prev_positions[prev_positions != 0].index)
    elif prev_positions is None:
        prev_positions = set()
    
    # Get alpha scores (required)
    alpha = signals.get("alpha")
    if alpha is None or len(alpha) == 0:
        return prev_positions.copy()  # No signals, keep current positions
    
    # Get entry/exit signals (optional, fallback to alpha)
    entry_signal = signals.get("entry", alpha)
    exit_signal = signals.get("exit", alpha)
    
    # Backward compatibility: if entry_quantile/exit_quantile provided, use for long only
    if entry_quantile is not None and entry_quantile_long is None:
        entry_quantile_long = entry_quantile
    if exit_quantile is not None and exit_quantile_long is None:
        exit_quantile_long = exit_quantile
    
    # Start with previous positions
    new_positions = prev_positions.copy()
    
    # Apply exit logic first (optimized: use vectorized operations)
    # Handle long positions exit (only for positions with positive alpha)
    if exit_quantile_long is not None:
        # Quantile-based exit for long positions
        exit_threshold_val_long = exit_signal.quantile(exit_quantile_long)
        positions_in_signal = [a for a in new_positions if a in exit_signal.index]
        if positions_in_signal:
            # Only check long exit for positions where alpha suggests it's a long position
            long_positions = [a for a in positions_in_signal if exit_signal[a] >= 0]
            if long_positions:
                exit_mask_long = exit_signal[long_positions] < exit_threshold_val_long
                assets_to_exit_long = set(exit_signal[long_positions][exit_mask_long].index)
                new_positions -= assets_to_exit_long
    elif exit_quantile is not None:
        # Backward compatibility: quantile-based exit (long only)
        exit_threshold_val = exit_signal.quantile(exit_quantile)
        positions_in_signal = [a for a in new_positions if a in exit_signal.index]
        if positions_in_signal:
            exit_mask = exit_signal[positions_in_signal] < exit_threshold_val
            assets_to_exit = set(exit_signal[positions_in_signal][exit_mask].index)
            new_positions -= assets_to_exit
    elif exit_threshold is not None:
        # Threshold-based exit
        positions_in_signal = [a for a in new_positions if a in exit_signal.index]
        if positions_in_signal:
            exit_mask = exit_signal[positions_in_signal] < exit_threshold
            assets_to_exit = set(exit_signal[positions_in_signal][exit_mask].index)
            new_positions -= assets_to_exit
    
    # Handle short positions exit (only for positions with negative alpha)
    if exit_quantile_short is not None:
        # Quantile-based exit for short positions
        exit_threshold_val_short = exit_signal.quantile(exit_quantile_short)
        positions_in_signal = [a for a in new_positions if a in exit_signal.index]
        if positions_in_signal:
            # Only check short exit for positions where alpha suggests it's a short position
            short_positions = [a for a in positions_in_signal if exit_signal[a] <= 0]
            if short_positions:
                exit_mask_short = exit_signal[short_positions] > exit_threshold_val_short
                assets_to_exit_short = set(exit_signal[short_positions][exit_mask_short].index)
                new_positions -= assets_to_exit_short
    
    # Apply entry logic
    # Handle long positions entry
    if entry_quantile_long is not None:
        # Quantile-based entry for long positions
        entry_threshold_val_long = entry_signal.quantile(entry_quantile_long)
        entry_candidates_long = set(entry_signal[entry_signal >= entry_threshold_val_long].index)
        new_positions.update(entry_candidates_long)
    elif entry_quantile is not None:
        # Backward compatibility: quantile-based entry (long only)
        entry_threshold_val = entry_signal.quantile(entry_quantile)
        entry_candidates = set(entry_signal[entry_signal >= entry_threshold_val].index)
        new_positions.update(entry_candidates)
    elif entry_threshold is not None:
        # Threshold-based entry
        entry_candidates = set(entry_signal[entry_signal >= entry_threshold].index)
        new_positions.update(entry_candidates)
    
    # Handle short positions entry
    if entry_quantile_short is not None:
        # Quantile-based entry for short positions
        entry_threshold_val_short = entry_signal.quantile(entry_quantile_short)
        entry_candidates_short = set(entry_signal[entry_signal <= entry_threshold_val_short].index)
        new_positions.update(entry_candidates_short)
    
    # Default entry logic (if no explicit entry logic provided)
    if (entry_quantile_long is None and entry_quantile_short is None and 
        entry_quantile is None and entry_threshold is None):
        # Default: use all assets with positive alpha if no entry logic provided
        # This ensures positions are created when no explicit entry logic is specified
        entry_candidates = set(entry_signal[entry_signal > 0].index)
        if len(entry_candidates) == 0:
            # If no positive signals, use top 50% by default
            median_val = entry_signal.median()
            entry_candidates = set(entry_signal[entry_signal >= median_val].index)
        new_positions.update(entry_candidates)
    
    return new_positions


def _get_model_signals(model, train_data, prev_positions, t, model_accepts_t, valid_assets):
    """Helper to execute model and validate signals."""
    if model_accepts_t:
        signals = model(train_data, prev_positions, pd.Timestamp(t))
    else:
        signals = model(train_data, prev_positions)
    
    if not isinstance(signals, dict):
        raise TypeError(f"`model` must return a dict with signals, got {type(signals)}")
    if "alpha" not in signals:
        raise ValueError("`model` must return dict with 'alpha' key")
        
    # Filter/Reindex to valid assets
    signals_filtered = {}
    for key, value in signals.items():
        if isinstance(value, pd.Series):
            signals_filtered[key] = value.reindex(valid_assets).fillna(0.0)
        else:
            signals_filtered[key] = value
            
    return signals_filtered


def _run_allocation_pipeline(prev_positions, signals, position_kwargs, sizing_engine, prices_history):
    """Helper to execute position and sizing engines."""
    # 1. Position Engine
    prev_pos_set = prev_positions.copy() if isinstance(prev_positions, set) else set(prev_positions) if prev_positions is not None else set()
    new_positions = position_engine(prev_pos_set, signals, **position_kwargs)
    
    # 2. Sizing Engine
    weights = sizing_engine(new_positions, signals, prices_history)
    
    return new_positions, weights



def oos_walk_forward_backtest(
    prices: pd.Series | pd.DataFrame,
    model: Callable[..., dict],
    *,
    refit_schedule: str | None = None,
    rebalance_schedule: str | None = None,
    train_window: int = 252 * 2,
    window_type: str = "expanding",
    min_assets: int = 20,
    target_assets: int = 40,
    min_obs_frac: float = 0.95,
    min_train_rows: int = 100,
    on_fit_error: str = "skip",
    freq: int = 252,
    lag: int = 1,
    risk_free_rate: float = 0.0,
    allow_all_cash: bool = False,
    transaction_cost: float = 0.0,
    baseline: str | None = None,
    n_jobs: int = 1,
    entry_quantile: float | None = None,
    exit_quantile: float | None = None,
    sizing_engine: Callable,
    position_engine_kwargs: dict | None = None,
    debug: bool = False,
    oos_performance_callback: Callable | None = None,
) -> dict:
    """
    Out-of-sample walk-forward pipeline with signal-based architecture.

    **Dynamic Walk-Forward:**
    Separates model refitting (training) from portfolio rebalancing (trading).
    - `refit_schedule`: Controls when the model is retrained/updated.
    - `rebalance_schedule`: Controls when signals are checked and positions updated.
    
    This allows for setups like "train monthly, trade daily" or "train weekly, trade weekly".
    Transaction costs are applied whenever weights change.

    **Model Function:**
    `model` must return a dict with signal components:
    {
        "alpha": pd.Series,      # Cross-sectional alpha scores (index=assets, values=scores)
        "entry": pd.Series,       # Optional: Entry signals (boolean or scores)
        "exit": pd.Series,        # Optional: Exit signals (boolean or scores)
        "confidence": pd.Series  # Optional: Confidence scores
    }
    
    Model signature: `model(train_returns, previous_positions?, date?) -> dict`
    Model should NOT do: weight normalization, .shift(), exposure constraints, position tracking

    Args:
        prices: Price series/dataframe with DatetimeIndex
        model: Callable that returns signals dict. Signature: (train_returns, previous_positions?, date?)
        refit_schedule: When to retrain the model (e.g., 'ME' for month-end, 'W' for weekly).
        rebalance_schedule: When to check entry/exit in dynamic mode (e.g., 'D' for daily).
                           If None, defaults to `refit_schedule`.
        train_window: Number of days of history to use for training (default: 504 days = 2 years).
                      In 'rolling' mode, this is the window size.
                      In 'expanding' mode, this is the minimum warmup period.
        window_type: 'rolling' or 'expanding' (default).
                     'rolling': Uses a fixed window size defined by train_window.
                     'expanding': Uses all available history starting from the beginning.
        rebalance: [DEPRECATED] Use `refit_schedule` instead.
        min_assets: Minimum number of assets required to generate signals
        target_assets: Maximum number of assets to include in portfolio
        min_obs_frac: Minimum fraction of non-NaN observations required per asset (default: 0.95)
        min_train_rows: Minimum number of training observations required
        on_fit_error: How to handle model errors: 'skip' or 'raise'
        freq: Annual frequency for return calculations (default: 252 for daily data)
        lag: Lag in days between signal and execution (default: 1). Applied internally by backtest().
            Signal Close t → Trade Close t+lag → Capture Return Close t+lag to Close t+lag+1.
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        allow_all_cash: If True, allows portfolio with no positions
        transaction_cost: Transaction cost as fraction of traded notional (applied when weights change)
        baseline: Baseline strategy name for comparison (e.g., 'equal_weight')
        n_jobs: Number of parallel jobs for batch prediction generation in dynamic mode.
                Defaults to 1. Set to -1 to use all available cores.
        entry_quantile: [DEPRECATED] Quantile threshold for entry. Use position_engine_kwargs instead.
        exit_quantile: [DEPRECATED] Quantile threshold for exit. Use position_engine_kwargs instead.
        sizing_engine: Callable that converts positions to weights. REQUIRED.
                      Signature: (positions, signals, prices, **kwargs) -> pd.Series
                      Example: Equal-weight sizing: `lambda pos, sig, px, **kw: pd.Series(1.0/len(pos), index=sorted(pos)) if len(pos) > 0 else pd.Series(dtype=float)`
        position_engine_kwargs: Dict of kwargs for position_engine(). Can include:
                               entry_threshold, exit_threshold, entry_quantile, exit_quantile,
                               min_hold_periods, cooldown_periods
        debug: If True, print debug information
        oos_performance_callback: Optional callback function called after each OOS period completes.
                                 Signature: (refit_date, sharpe, returns, equity) -> None
                                 Used for incremental OOS performance tracking for meta-learning.
                                 Called with OOS performance data for each refit period as it completes.

    Returns:
        dict: Backtest results including:
            - 'equity': Equity curve (pd.Series)
            - 'returns': Strategy returns (pd.Series)
            - 'sharpe': Sharpe ratio
            - 'ann_return': Annualized return
            - 'ann_vol': Annualized volatility
            - 'max_drawdown': Maximum drawdown
            - 'weights': Portfolio weights (pd.DataFrame)
            - 'wf': Walk-forward metadata dict with 'fit_dates', 'first_active', 'mode'
            - ... and other backtest metrics
    """
    px = _as_prices_df(prices).sort_index()
    returns = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    if window_type not in ("rolling", "expanding"):
        raise ValueError(f"`window_type` must be 'rolling' or 'expanding', got {window_type}")

    if refit_schedule is None:
        raise ValueError("`refit_schedule` must be provided (e.g., 'ME', 'W').")

    # Default: rebalance_schedule same as refit_schedule
    if rebalance_schedule is None:
        rebalance_schedule = refit_schedule

    # Walk-forward weights generation
    idx = returns.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("`prices` index must be a DatetimeIndex for walk-forward backtest.")

    # Get dates for refitting and rebalancing
    refit_dates = _rebalance_dates(idx, refit_schedule)
    rebal_dates = _rebalance_dates(idx, rebalance_schedule)
    refit_dates_set = set(refit_dates)
    
    # Process union of refit and rebalance dates
    # We need to run the pipeline on both:
    # - Refit dates: to update the model
    # - Rebalance dates: to check signals and update positions
    dates_to_process = sorted(list(set(refit_dates) | set(rebal_dates)))

    if debug:
        print(f"[DEBUG] Dates DEBUG:", file=sys.stderr)
        print(f"  len(refit_dates): {len(refit_dates)} (Schedule: {refit_schedule})", file=sys.stderr)
        print(f"  len(rebal_dates): {len(rebal_dates)} (Schedule: {rebalance_schedule})", file=sys.stderr)
        print(f"  len(dates_to_process): {len(dates_to_process)}", file=sys.stderr)

    if len(dates_to_process) == 0:
        raise ValueError("No rebalance dates found. Check schedule parameters and data index.")
    
    # Check if model accepts rebalance date as third argument
    sig = inspect.signature(model)
    model_accepts_t = len(sig.parameters) >= 3
    
    # Set up position engine kwargs (merge with deprecated entry/exit_quantile for backward compatibility)
    if position_engine_kwargs is None:
        position_engine_kwargs = {}
    # Backward compatibility: use entry_quantile/exit_quantile if provided
    if entry_quantile is not None and "entry_quantile" not in position_engine_kwargs:
        position_engine_kwargs["entry_quantile"] = entry_quantile
    if exit_quantile is not None and "exit_quantile" not in position_engine_kwargs:
        position_engine_kwargs["exit_quantile"] = exit_quantile
    
    # Initialize weights DataFrame (desired weights, before lag)
    # Use NaN instead of 0.0 so ffill works correctly in interval mode
    desired_weights = pd.DataFrame(np.nan, index=idx, columns=returns.columns)
    fit_dates = []
    executed_dates = []  # Track dates where we actually generated weights
    oos_performance_data = {}
    prev_positions = set()  # Track positions, not weights
    
    # State variables for the loop
    current_signals = None
    current_valid_assets = None
    last_refit_date = None
    
    # Track how many dates get weights (for debug output)
    dates_with_weights = 0
    
    # Unified processing loop
    for t in dates_to_process:
        if t not in idx:
            continue
            
        # 1. Refit Model / Generate Signals
        # We generate signals on all processing dates to ensure fresh data for entry/exit checks
        # But we only track 'refit' events for OOS callbacks and metadata if t is in refit_dates_set
        t_pos = idx.get_loc(t)
        if t_pos < train_window:
            continue

        if window_type == "expanding":
            train_start = 0
        else:
            train_start = max(0, t_pos - train_window)
        train_end = t_pos
        train = returns.iloc[train_start:train_end]
        
        if len(train) < min_train_rows:
            continue
        
        # Calculate OOS performance for previous period (Meta-Learning)
        # Only trigger on refit dates to maintain consistent OOS periods
        if t in refit_dates_set and last_refit_date is not None and oos_performance_callback is not None:
            # Define OOS mask on the index directly
            oos_mask = (idx > last_refit_date) & (idx <= t)
            oos_dates = idx[oos_mask]
            
            if len(oos_dates) > 0:
                # Extract weights for this period from desired_weights
                w_slice = desired_weights.loc[oos_dates]
                w_slice = w_slice.fillna(0.0)
                
                p_slice = px.loc[oos_dates]
                common_cols = w_slice.columns.intersection(p_slice.columns)
                
                if len(common_cols) > 0:
                    try:
                        # Mini-backtest for OOS metrics
                        oos_res = backtest(
                            weights=w_slice[common_cols],
                            prices=p_slice[common_cols],
                            freq=freq, lag=lag, risk_free_rate=risk_free_rate,
                            transaction_cost=transaction_cost, compute_risk_metrics=False
                        )
                        
                        if oos_res.get("returns") is not None:
                            r = oos_res["returns"]
                            sharpe = r.mean() / r.std() * np.sqrt(freq) if len(r) > 1 and r.std() > 0 else np.nan
                            
                            oos_performance_data[last_refit_date] = True # Mark as processed
                            oos_performance_callback(last_refit_date, sharpe, r, oos_res["equity"])
                    except Exception:
                        pass # Skip if metrics fail

        # Asset Selection
        obs_counts = train.notna().sum()
        min_obs = int(len(train) * min_obs_frac)
        valid_assets = obs_counts[obs_counts >= min_obs].index
        
        if len(valid_assets) < min_assets:
            continue
        
        if len(valid_assets) > target_assets:
            coverage = obs_counts[valid_assets] / len(train)
            valid_assets = coverage.nlargest(target_assets).index
        
        train_sub = train[valid_assets]
        
        try:
            # Generate Signals
            current_signals = _get_model_signals(
                model, train_sub, prev_positions, t, model_accepts_t, valid_assets
            )
            current_valid_assets = valid_assets
            
            if t in refit_dates_set:
                fit_dates.append(t)
                last_refit_date = t
            
        except Exception as e:
            if debug:
                print(f"[DEBUG] Fit Error at {t}: {e}", file=sys.stderr)
            if on_fit_error == "skip":
                continue
            elif on_fit_error == "raise":
                raise
            else:
                raise ValueError(f"`on_fit_error` must be 'skip' or 'raise', got {on_fit_error}")

        # 2. Allocate Weights (if we have valid signals)
        if current_signals is not None:
            try:
                prev_positions, weights_t = _run_allocation_pipeline(
                    prev_positions, 
                    current_signals, 
                    position_engine_kwargs,
                    sizing_engine,
                    px.loc[:t]  # Pass full history up to t
                )
                
                if len(weights_t) > 0:
                    weights_t = weights_t.reindex(current_valid_assets, fill_value=0.0)
                    desired_weights.loc[t, weights_t.index] = weights_t
                    executed_dates.append(t)
                
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Exception processing {t}: {type(e).__name__}: {e}", file=sys.stderr)

    # No forward-fill, keep zeros between check dates (drift handles the rest)
    desired_weights = desired_weights.fillna(0.0)

    # FIX: Don't apply execution lag here - let backtest() handle it internally
    # This avoids double-shifting and index misalignment issues
    # The desired_weights are indexed by signal dates, and backtest() will
    # apply the lag by reading weights from signal dates and applying them
    # at execution dates (signal_date + lag + 1)
    weights_oos_full = desired_weights.fillna(0.0)

    # DEBUG: Final check before active array
    if debug:
        print(f"[DEBUG] Final Check (weights_oos_full):", file=sys.stderr)
        active_sum = weights_oos_full.abs().sum(axis=1)
        active_count = (active_sum > 0).sum()
        print(f"  Active dates (sum > 0): {active_count}/{len(weights_oos_full)}", file=sys.stderr)
        print(f"  Total weight sums range: [{active_sum.min():.6f}, {active_sum.max():.6f}]", file=sys.stderr)
        if active_count == 0:
            print(f"  ERROR: No active dates found! This will trigger the ValueError.", file=sys.stderr)
            print(f"  Fit dates processed: {len(fit_dates)}", file=sys.stderr)
            if len(fit_dates) > 0:
                print(f"  First fit date: {fit_dates[0]}, Last fit date: {fit_dates[-1]}", file=sys.stderr)

    # Find first active date (before lag application, since backtest will apply it)
    active = weights_oos_full.abs().sum(axis=1) > 0
    if not active.any():
        if not allow_all_cash:
            raise ValueError("No non-zero weights were produced; adjust window/coverage constraints.")
        weights_oos = weights_oos_full
        prices_oos = px.loc[:, weights_oos.columns]
        first_active = prices_oos.index[0]
    else:
        first_active = active.idxmax()
        weights_oos = weights_oos_full.loc[first_active:]
        prices_oos = px.loc[first_active:, weights_oos.columns]

    # Prepare backtest arguments
    # Let backtest() handle lag application internally
    # Plotting is now handled by results_backtest() function, not here
    backtest_kwargs = {
        "weights": weights_oos,
        "prices": prices_oos,
        "freq": freq,
        "lag": lag,  # Let backtest() handle lag application
        "risk_free_rate": risk_free_rate,
        "transaction_cost": transaction_cost,
        "compute_risk_metrics": True,  # Can be set to False for speedup if not needed
        "baseline": baseline,
    }

    # Pass signal_dates to control when rebalancing occurs.
    # We use executed_dates (all dates where we successfully generated weights),
    # not just fit_dates, to allow for dynamic rebalancing between refits.
    # backtest() will handle shifting these by lag internally.
    if executed_dates:
        backtest_kwargs["signal_dates"] = executed_dates
    else:
        backtest_kwargs["signal_dates"] = None

    res = backtest(**backtest_kwargs)
    res["weights"] = weights_oos
    res["wf"] = {
        "fit_dates": fit_dates,
        "first_active": first_active,
    }
    
    # Final OOS performance tracking for meta-learning
    # Also calculate for the last refit period (from last refit date to end of data)
    # This ensures consistency with the original approach
    if oos_performance_callback is not None and len(fit_dates) > 0:
        returns = res.get("returns")
        equity = res.get("equity")
        
        if returns is not None and equity is not None:
            # Skip intermediate periods - they were already calculated incrementally above
            # Only handle the last refit period if it wasn't already processed
            # (The last period is from last refit date to end of data)
            if len(fit_dates) > 0:
                last_refit_date = fit_dates[-1]
                
                # OOS period is from last refit date to end of data
                # Use shared function for consistency with hyperparameter optimization
                # For the last period, we use the last date in returns as the end boundary
                if len(returns.index) > 0:
                    last_returns_date = returns.index[-1]
                    oos_dates = _get_oos_period_boundaries(
                        last_refit_date, 
                        last_returns_date, 
                        returns.index
                    )
                else:
                    oos_dates = pd.DatetimeIndex([])
                
                if len(oos_dates) > 0:
                    oos_returns = returns.loc[oos_dates]
                    oos_equity = equity.loc[oos_dates] if equity is not None else (1 + oos_returns).cumprod()
                    
                    # Calculate Sharpe for this OOS period
                    if len(oos_returns) > 1 and oos_returns.std() > 0:
                        oos_sharpe = oos_returns.mean() / oos_returns.std() * np.sqrt(freq)
                    else:
                        oos_sharpe = np.nan
                    
                    # Only call callback if not already called incrementally
                    if last_refit_date not in oos_performance_data:
                        oos_performance_callback(
                            refit_date=last_refit_date,
                            sharpe=oos_sharpe,
                            returns=oos_returns,
                            equity=oos_equity
                        )
    
    return res

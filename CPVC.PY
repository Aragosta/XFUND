# BACKTEST

#  CPCV ALGORITHM , may be faulty

import numpy as np
import pandas as pd
import itertools
import KFOLD

def cpcvImplementation(X, y, clf, num_folds, k, scoring):
    T = X.shape[0]
    groups = np.array_split(range(T), num_folds)
    combinations = itertools.combinations(range(num_folds), k)
    scores = []

    for testing_groups in combinations:
        training_groups = set(range(num_folds)) - set(testing_groups)
        training_indices = np.hstack([groups[i] for i in training_groups])
        testing_indices = np.hstack([groups[i] for i in testing_groups])

        t1_train = t1.iloc[training_indices]
        cvGen = KFOLD.PurgedKFold(n_splits=len(training_groups), t1=t1_train, pctEmbargo=0.01)  # Adjust pctEmbargo as needed

        # Purge and Embargo
        train_indices_purged, _ = next(cvGen.split(X.iloc[training_indices]))
        
        # Fit and Score
        clf.fit(X.iloc[train_indices_purged], y.iloc[train_indices_purged])
        score = clf.score(X.iloc[testing_indices], y.iloc[testing_indices])  # Change to custom scoring if needed
        scores.append(score)

    return np.mean(scores), np.std(scores)


def calculate_metrics(predictions, prices, risk_free_rate=0.0):
    """
    Calculate Sharpe Ratio, Returns, and Maximum Drawdown.
    predictions: Model predictions corresponding to market positions (1, -1, or 0)
    prices: Actual market prices (used to calculate returns)
    risk_free_rate: Risk-free rate for Sharpe Ratio calculation
    """
    # Convert predictions to market positions and calculate daily returns
    positions = predictions.shift(1)  # Shift positions to avoid look-ahead bias
    daily_returns = positions * (prices.pct_change())

    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod()

    # Calculate maximum drawdown
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # Calculate Sharpe Ratio
    excess_returns = daily_returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    return {
        'cumulative_returns': cumulative_returns[-1],
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }


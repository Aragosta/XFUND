import pandas as pd
import numpy as np



def getTEvents(graw,h):
    tEvents, sPos, Sneg = [], 0, 0
    diff = graw.diff()
    for i in diff.index[1:]:
        sPos, Sneg = max(0, sPos+diff.loc[i]), min(0, Sneg+diff.loc[i])
        if Sneg<-h:
            Sneg=0; tEvents.append(i)
        elif sPos>h:
            sPos=0; tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def getDailyVol(close, span0=100):
    """
    Compute daily volatility adjusted for auto-correlation

    Args:
    close (pd.Series): Closing prices
    span0 (int): Span parameter for Exponential Weighted Moving Average (EWMA)

    Returns:
    pd.Series: Daily volatility, reindexed to match the original close series
    """
    # Find the timestamp of the previous trading day for each trading day
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])

    # Compute daily returns
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1 

    # Apply Exponential Weighted Moving Standard Deviation
    df0 = df0.ewm(span=span0).std()

    return df0


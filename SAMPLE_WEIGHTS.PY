import pandas as pd
import numpy as np


def num_conc_events(closeIdx: np.ndarray, t1: pd.Series, molecule: np.ndarray) -> pd.Series:
    '''
    Computing the number of concurrent events per bar
    
        Parameters:
            closeIdx (np.ndarray): timestamps of close prices
            t1 (pd.Series): series with the timestamps of the vertical barriers
            molecule (np.ndarray): dates of events on which weights are computed
            
        Returns:
            pd.Series with number of labels concurrent at each timestamp
    '''
    t1 = t1.fillna(closeIdx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]
    iloc = closeIdx.searchsorted(pd.DatetimeIndex([t1.index[0], t1.max()]))
    count = pd.Series([0] * (iloc[1] + 1 - iloc[0]), index=closeIdx[iloc[0]: iloc[1] + 1])
    for tIn, tOut in t1.iteritems():
        count.loc[tIn: tOut] += 1
    return count.loc[molecule[0]: t1[molecule].max()]
def sample_weights(t1: pd.Series, num_conc_events: pd.Series, molecule: np.ndarray) -> pd.Series:
    '''
    Computing average uniqueness over the event's lifespan
    
        Parameters:
            t1 (pd.Series): series with the timestamps of the vertical barriers
            num_conc_events (pd.Series): number of concurrent events per bar
            molecule (np.ndarray): dates of events on which weights are computed
            
        Returns:
            weights (pd.Series): weights that represent the average uniqueness
    '''
    weights = pd.Series([0] * len(molecule), index=molecule)
    for tIn, tOut in t1.loc[weights.index].iteritems():
        weights.loc[tIn] = (1.0 / num_conc_events.loc[tIn: tOut]).mean()
    return weights
def get_ind_matrix(barIdx: np.ndarray, t1: pd.Series) -> pd.DataFrame:
    '''
    Deriving indicator matrix
    
        Parameters:
            barIdx (np.ndarray): indexes of bars
            t1 (pd.Series): series with the timestamps of the vertical barriers
            
        Returns:
            indM (pd.DataFrame): binary matrix indicating what bars influence the label for each observation
    '''
    indM = pd.DataFrame(0, index=barIdx, columns=range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1, i] = 1.0
    return indM


def get_avg_uniqueness(indM: pd.DataFrame) -> float:
    '''
    Compute average uniqueness from indicator matrix
    '''
    c = indM.sum(axis=1)
    u = indM.div(c, axis=0)
    avg_uniq = u[u > 0].mean()
    return avg_uniq


def seq_bootstrap(indM: pd.DataFrame, sLength: int = None) -> np.ndarray:
    '''
    Generate a sample via sequential bootstrap
    
        Parameters:
            indM (pd.DataFrame): binary matrix indicating what bars influence the label for each observation
            sLength (int) (optional): sample length (if None, equals number of columns in indM)
            
        Returns:
            phi (np.ndarray): array with indexes of the features sampled by sequential bootstrap
    '''
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avg_uniq = pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]]
            avg_uniq.loc[i] = get_avg_uniqueness(indM_).iloc[-1]
        prob = avg_uniq / avg_uniq.sum()
        phi += [np.random.choice(indM.columns, p=prob)]
    return np.array(phi)

#monte carlo

def gen_rand_t1(numObs: int, numBars: int, maxH: int) -> pd.Series:
    '''
    Generate random t1 series
    
        Parameters:
            numObs (int): number of observations for which t1 is generated
            numBars (int): number of bars
            maxH (int): upper bound for uniform distribution to determine the number of bars spanned by observation
        Returns:
            t1 (pd.Series)
    '''
    t1 = pd.Series()
    for i in range(numObs):
        idx = np.random.randint(0, numBars)
        val = idx + np.random.randint(1, maxH)
        t1.loc[idx] = val
    return t1.sort_index()


def aux_MC(numObs: int, numBars: int, maxH: int) -> dict:
    '''
    Generate random t1 series
    
        Parameters:
            numObs (int): number of observations for which t1 is generated
            numBars (int): number of bars
            maxH (int): upper bound for uniform distribution to determine the number of bars spanned by observation
        Returns:
            dict with average uniqueness derived by standard and sequential bootstrap algorithms
    '''
    t1 = gen_rand_t1(numObs, numBars, maxH)
    barIdx = range(t1.max() + 1)
    indM = get_ind_matrix(barIdx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = get_avg_uniqueness(indM[phi]).mean()
    phi = seq_bootstrap(indM)
    seqU = get_avg_uniqueness(indM[phi]).mean()
    return {'stdU': stdU, 'seqU': seqU}


def main_MC(numObs: int, numBars: int, maxH: int, numIters: int) -> None:
    '''
    Run MC simulation for comparing standard and sequential bootstraps
    
        Parameters:
            numObs (int): number of observations for which t1 is generated
            numBars (int): number of bars
            maxH (int): upper bound for uniform distribution to determine the number of bars spanned by observation
            numIters (int): number of MC iterations
        Returns:
            out (pd.DataFrame): dataframe containing uniqueness obtained by standard and sequential bootstraps
    '''
    out = pd.DataFrame()
    for i in range(numIters):
        out = pd.concat((out, pd.DataFrame([aux_MC(numObs, numBars, maxH)])))
    return out
def sample_return_weights(
    t1: pd.Series, num_conc_events: pd.Series, close: pd.Series, molecule: np.ndarray
) -> pd.Series:
    '''
     Determination of sample weights by absolute return distribution
    
        Parameters:
            t1 (pd.Series): series with the timestamps of the vertical barriers
            num_conc_events (pd.Series): number of concurrent events per bar
            close (pd.Series): close prices
            molecule (np.ndarray): dates of events on which weights are computed
            
        Returns:
            weights (pd.Series): weights that absolute returns
    '''
    ret = np.log(close).diff()
    weights = pd.Series(index=molecule, dtype=object)
    for tIn, tOut in t1.loc[weights.index].iteritems():
        weights.loc[tIn] = (ret.loc[tIn: tOut] / num_conc_events.loc[tIn: tOut]).sum()
    return weights.abs()
def get_time_decay(tW: pd.Series, clfLastW: float = 1.0) -> pd.Series:
    '''
    Apply piecewise-linear decay to observed uniqueness. Newest observation gets weight=1,
    oldest observation gets weight=clfLastW.
    
        Parameters:
            tW (pd.Series): observed uniqueness
            clfLastW (float): weight for the oldest observation
        
        Returns:
            clfW (pd.Series): series with time-decay factors
    '''
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1. / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1.0 - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    # print(const, slope)
    return clfW
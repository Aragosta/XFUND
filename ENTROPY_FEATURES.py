# USEFULL ENTROPY FEATURES 

import time
import numpy as np
from typing import Any, Optional, Tuple

# Plug-in entropy estimator

def plugin(msg, w):
    # Compute plug-in entropy rate
    pmf = pmf1(msg, w)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf])
    return out, pmf

def pmf1(msg: Any, w: int) -> dict:
    '''
    Computes the probability mass function for a one-dimensional random variable 
    (len(msg) - w occurrences).
    
    Parameters:
        msg (Any): Sequence with observations (usually a string)
        w (int): Word length used for PMF estimation
        
    Returns:
        pmf (dict): Dictionary with words as keys and their estimated probabilities as values.
    '''
    lib = {}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    for i in range(w, len(msg)):
        msg_ = msg[i - w: i]
        if msg_ not in lib:
            lib[msg_] = [i - w]
        else:
            lib[msg_].append(i - w)
    length = float(len(msg) - w)
    pmf = {i: len(lib[i]) / length for i in lib}
    return pmf

def lempel_ziv_lib(msg: str) -> list:
    '''
    Implements the LZ algorithm to construct a library of unique substrings.
    
    Parameters:
        msg (str): Sequence with observations
        
    Returns:
        lib (list): List containing unique substrings
    '''
    i, lib = 1, [msg[0]]
    while i < len(msg):
        for j in range(i, len(msg)):
            msg_ = msg[i: j + 1]
            if msg_ not in lib:
                lib.append(msg_)
                break
        i = j + 1
    return lib

def match_length(msg: str, i: int, n: int) -> Tuple[int, str]:
    '''
    Computes the length of the longest match.
    
    Parameters:
        msg (str): Sequence with observations
        i (int): Position before which we look for a match
        n (int): Size of the window for searching for a match
        
    Returns:
        len(subS) + 1 (int): Length of the match + 1
        subS (str): Matched substring
    '''
    subS = ''
    for l in range(n):
        msg1 = msg[i: i + 1 + l]
        for j in range(i - n, i):
            msg0 = msg[j: j + 1 + l]
            if msg1 == msg0:
                subS = msg1
                break
    return len(subS) + 1, subS

def konto(msg: Any, window: Optional[int] = None) -> dict:
    '''
    Kontoyiannis' LZ entropy estimate, 2013 version (centered window). Inverse of the average length 
    of the shortest non-redundant substring. If non-redundant substrings are short, the text is 
    highly entropic. window=None for expanding window, in which case len(msg) % 2 = 0. If the end 
    of the message is more relevant, try konto(msg[::-1]).
    
    Parameters:
        msg (Any): Sequence with observations (usually a string)
        window (Optional[int]): Window size for constant window
        
    Returns:
        out (dict): Dictionary with results
    '''
    out = {'num': 0, 'sum': 0, 'subS': []}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    if window is None:
        points = range(1, len(msg) // 2 + 1)
    else:
        window = min(window, len(msg) // 2)
        points = range(window, len(msg) - window + 1)
    for i in points:
        if window is None:
            l, subS = match_length(msg, i, i)
            out['sum'] += np.log2(i + 1) / l  # to avoid Doeblin condition
        else:
            l, subS = match_length(msg, i, window)
            out['sum'] += np.log2(window + 1) / l  # to avoid Doeblin condition
        out['subS'].append(subS)
        out['num'] += 1
    out['h'] = out['sum'] / out['num']
    out['r'] = 1 - out['h'] / np.log2(len(msg))  # redundancy, 0 <= r <= 1
    return out


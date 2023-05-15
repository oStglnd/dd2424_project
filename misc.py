
import numpy as np
import pickle
import scipy.io as sio

def softMax(S: np.array) -> np.array:
    """
    Parameters
    ----------
    S : dxN score matrix

    Returns
    -------
    S : dxN score matrix w. applied softmax activation
    """
    S = np.exp(S)
    return S / np.sum(S, axis=0)

def oneHotEncode(k: np.array) -> np.array:
    """
    Parameters
    ----------
    k : Nx1 label vector

    Returns
    -------
    Y: NxK one-hot encoded label matrix
    """
    numCats = np.max(k)
    return np.array([[
        1 if idx == label else 0 for idx in range(numCats+1)]
         for label in k]
    )
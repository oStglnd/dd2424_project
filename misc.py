
import numpy as np
import pickle
import scipy.io as sio


def readData(fpath: str) -> object:

    with open(fpath, 'r') as fo:
        data = fo.read()

    # with open(fpath, 'r') as fo:
    #     data = fo.readlines()

    # # split lines into words and words into chars
    # data = [char
    #             for line in data
    #                 for word in list(line)
    #                     for char in list(word)
    # ]

    return data


def prepareData(data: object) -> dict:

    uniqueChars = set(data)
    keyToChar = dict(enumerate(uniqueChars))
    charToKey = dict([(val, key) for key, val in keyToChar.items()])

    return keyToChar, charToKey


def generateSequences(data: np.array, seq_length: int) -> np.array:
    x = []
    for i in range(len(data) - seq_length - 1):
        x.append(data[i:i+seq_length+1])

    return x


def sigmoid(S: np.array) -> np.array:
    """
    Parameters
    ----------
    S : dxN score matrix

    Returns
    -------
    S : dxN score matrix w. applied sigmoid activation
    """
    return 1 / (1 + np.exp(-S))


def softMax(S: np.array, temperature: float) -> np.array:
    """
    Parameters
    ----------
    S : dxN score matrix
    T : Scales variance in output probability distribution

    Returns
    -------
    S : dxN score matrix w. applied softmax activation
    """
    S = S / temperature
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

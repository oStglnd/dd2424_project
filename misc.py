
import numpy as np
import pickle
import scipy.io as sio
import string


def readData(fpath: str) -> object:

    with open(fpath, 'r') as fo:
        data = fo.read()

    return data

# turn a doc into clean tokens


def cleanData(doc):

    # replace '--' with a space ' '
    data = doc.replace('--', ' ')

    # split into tokens by whitespace
    tokens = data.split()

    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]

    # remove non alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]

    # make all tokens lower case
    tokens = [word.lower() for word in tokens]

    return tokens


def prepareData(data: object) -> dict:

    uniqueChars = set(data)
    keyToChar = dict(enumerate(uniqueChars))
    # keyToChar = dict(enumerate(np.unique(data)))
    charToKey = dict([(val, key) for key, val in keyToChar.items()])

    return keyToChar, charToKey


def generateSequences(data: np.array, seq_length: int) -> np.array:
    x = []
    y = []
    for i in range(0, len(data) - seq_length - 1, seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+1:i+seq_length+1])

    return x, y


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


def softMax(S: np.array, temperature=1.0) -> np.array:
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

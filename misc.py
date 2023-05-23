
import numpy as np
import pickle
import scipy.io as sio
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import WordNetLemmatizer


def getCharData(fpath:str) -> list:
    # read text file
    with open(fpath, 'r') as fo:
        data = fo.readlines()
    
    # get sentences 
    sentences = ''.join(data).split('.')
    
    # get data container
    data = []
    
    # define charDrop list
    dropChars = ['3', '¤', '#', '&']
    for sentence in sentences:
        
        sentence = list(sentence)
        sentence = [char for char in sentence if char not in dropChars]
        
        data.append(sentence)
        
    return data


def getTextData(fpath: str) -> list:
    
    # read text file
    with open(fpath, 'r') as fo:
        data = fo.readlines()
    
    # get sentences 
    sentences = ''.join(data).split('.')
    
    data = []
    for sentence in sentences:
        # split lines into tokens
        sentence = word_tokenize(''.join(sentence))
        
        # keepList = ['.', ',', ':', ';', '\\n']
        keepList = []
        sentence = [word for word in sentence if word.isalpha() or word in keepList]
    
        # remove stopwords
        stops = stopwords.words('english') 
        sentence = [word for word in sentence if word not in stops]
        
        # stem words
        stemmer = SnowballStemmer('english')
        sentence = [stemmer.stem(word) for word in sentence]
        
        # # lemmatize words
        # wnl = WordNetLemmatizer()
        # sentence = [wnl.lemmatize(word.lower()) for word in sentence]
    
        data.append(sentence)
    
    return data

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
    # keyToChar = dict(enumerate(np.unique(data)))
    charToKey = dict([(val, key) for key, val in keyToChar.items()])

    return keyToChar, charToKey


def generateSequences(data: np.array, seq_length: int) -> np.array:
    X = []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:i+seq_length+1])
    
    train_frac = 0.9
    train_n = int(len(X) * train_frac)
    
    X_train = X[:train_n]
    X_val = X[train_n:]
    
    return X_train, X_val


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


def softMax(S: np.array, temperature = 1.0) -> np.array:
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

def oneHotEncode_v2(k: int, K: int) -> np.array:
    """
    Parameters
    ----------
    k : label
    K : category size

    Returns
    -------
    y: 1xK one-hot encoded label matrix
    """
    y = np.zeros(shape=(1, K))
    y[0, k] = 1
    
    return y

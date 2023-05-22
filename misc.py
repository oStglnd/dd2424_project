
import numpy as np
import pickle
import scipy.io as sio
import string
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


# stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def readData(fpath: str) -> object:

    with open(fpath, 'r') as fo:
        data = fo.read()

    return data

# turn a doc into clean tokens


def cleanData(doc, stopwords=False):

    # replace '--' with a space ' '
    data = doc.replace('--', ' ')
    data = doc.replace('-', ' ')

    # split into tokens by whitespace
    tokens = data.split()

    # remove punctuation from each token
    remove = string.punctuation
    remove = remove.replace("'", "")
    table = str.maketrans('', '', remove)
    tokens = [w.translate(table) for w in tokens]

    # remove non alphabetic tokens
    # tokens = [word for word in tokens if word.isalpha()]

    # remove digits
    tokens = [re.sub(r'\d', '', w) for w in tokens]
    # remove special chars at beginning and end of words
    tokens = [re.sub(r'^[^\w]+|[^\w]+$', '', w) for w in tokens]

    # make all tokens lower case
    tokens = [word.lower() for word in tokens]

    # remove stopwords
    if stopwords:
        stop_words.update(['ii', 'iii', 'vi', 'xi', 'iv'])
        tokens = [word for word in tokens if word not in stop_words]

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

    return np.array(x), np.array(y)


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


def oneHotEncode(k: np.array, K: int) -> np.array:
    """
    Parameters
    ----------
    k : Nx1 label vector

    Returns
    -------
    Y: NxK one-hot encoded label matrix
    """
    numCats = K
    return np.array([[
        1 if idx == label else 0 for idx in range(numCats)]
        for label in k]
    )


def generateTrainingData(sentences, window_size):
    data = {}
    for sentence in sentences:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    V = len(data)
    data = sorted(list(data.keys()))
    vocab = {}

    for i in range(len(data)):
        vocab[data[i]] = i

    # for i in range(len(words)):
    X_train = []
    y_train = []

    for sentence in sentences:
        for i in range(len(sentence)):
            center_word = [0 for x in range(V)]
            center_word[vocab[sentence[i]]] = 1
            context = [0 for x in range(V)]

            for j in range(i-window_size, i+window_size):
                if i != j and j >= 0 and j < len(sentence):
                    context[vocab[sentence[j]]] += 1
            X_train.append(center_word)
            y_train.append(context)

    return X_train, y_train

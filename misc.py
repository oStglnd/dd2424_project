
import numpy as np
import pickle
import scipy.io as sio
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import WordNetLemmatizer

import re
def getAllWords(d):
    text_filtered = re.sub(r'[^a-z ]', '', d.lower()).split(" ")
    text_filtered2 = []
    for c in text_filtered:
        if c != ' ':
            text_filtered2.append(c)
    text_filtered = text_filtered2
    word_dict = {}
    for i,w in enumerate(text_filtered):
        if w not in word_dict:
            word_dict[w] = [i]
        else:
            word_dict[w].append(i)
    
    return word_dict, text_filtered

def countCorrectWords(guess, truth):
    filtered_guess = re.sub(r'[^a-z ]', '', guess.lower()).split(" ")
    text_filtered2 = []
    for c in filtered_guess:
        if c != ' ':
            text_filtered2.append(c)
    filtered_guess = text_filtered2
    num_correct_words = sum([1 if w in truth else 0 for w in filtered_guess])
    total_words = len(filtered_guess)
    # res = "Valid words generated {} out of {}".format(num_correct_words, total_words)
    return num_correct_words/total_words
def bleu(guess, truth, word_dict):
    def match_seq_len(guess_word_array, truth_word_array):
        match_count = 0
        j = 0
        while guess_word_array[j] == truth_word_array[j] and j < len(guess_word_array)-1 and j <= len(truth_word_array)-1:
            match_count += 1
            j += 1
        return match_count
    
                
    filtered_guess = re.sub(r'[^a-z ]', '', guess.lower()).split(" ")
    filtered_guess2 = []
    for c in filtered_guess:
        if c != ' ':
            filtered_guess2.append(c)
    filtered_guess = filtered_guess2

    n_grams = []
    for i in range(len(filtered_guess)):
        n_gram_curr_word = 0
        guess_to_match = filtered_guess[i:]
        if filtered_guess[i] in word_dict:
            for m in word_dict[filtered_guess[i]]:
                truth_to_match = truth[m :]
                tmp = match_seq_len(guess_to_match, truth_to_match)
                if tmp > n_gram_curr_word:
                    n_gram_curr_word = tmp
            n_grams.append(n_gram_curr_word)
    return (np.sum(np.square(np.array(n_grams)))) / len(filtered_guess)

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


import os
import json
import numpy as np
import string
import pickle

import gensim.models as gsm

from misc import oneHotEncode_v2, getCharData, getAllWords, countCorrectWords, bleu
from model_v3 import LSTM

# get paths
# home_path = os.path.dirname(os.getcwd())
home_path = os.getcwd()
data_path = home_path + '/data/'
model_path = home_path + '/models/'
# results_path = home_path + '\\a4\\results\\'

# get text data
fname = 'shakespeare.txt'
fpath = data_path + fname

# get text data
data = getCharData(fpath)

# define model 
model = 'model_v300_e2'
w2v = 'model_v300_w2v'

# define params
seq_length = 100
embeddings = True
embedding_dim = 200
units = [500]

# load word2Vec
w2v_model = gsm.Word2Vec.load(model_path + model)
char_vecs = w2v_model.wv

# split data into words
data = [char
          for sentence in data
              for char in sentence
]

# create word-key-word mapping
keyToChar = dict(enumerate(np.unique(data)))
charToKey = dict([(val, key) for key, val in keyToChar.items()])
keyToVec = dict([(key, char_vecs.get_vector(char)) for key, char in keyToChar.items()])
charToVec = dict([(char, char_vecs.get_vector(char)) for _, char in keyToChar.items()])

# define Y, w. one-hot encoded representations
K = len(charToKey)
X = []
for word in data:
    if embeddings: 
        X.append(charToVec[word].astype('float64'))
    else:
        X.append(oneHotEncode_v2(charToKey[word], K).astype('int8'))


# load model
K_out  = len(keyToChar)
K_in = embeddings * embedding_dim + (1 - embeddings) * K_out
sigma = 0

# define model
recurrentNet = LSTM(
    K_in=K_in,
    K_out=K_out,
    units=units,
    sigma=sigma,
    optimizer='adagrad',
    embeddings=embeddings,
    seed=2)

# load weigths
with open(model_path + model, 'r') as fo:
    weights = pickle.load(fo)

# set layer weights
recurrentNet.layers = weights

# insert embeddings
if embeddings:
    recurrentNet.keyToVec = keyToVec
    
# generate text
sequence = recurrentNet.synthesizeText(
    x0=X[0], 
    n=1000,
    temperature=0.5,
    threshold=0.9
)

# convert to chars and print sequence
sequence = ''.join([keyToChar[key] for key in sequence])
print('\nGenerated sequence \n\n {}\n'.format(sequence))

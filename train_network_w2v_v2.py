
import os
import json
import numpy as np
import string
import pickle

import gensim.models as gsm
import gensim.downloader as api

from misc import oneHotEncode_v2, getTextData, getCharData
from model_v3 import LSTM

def trainNetwork(
        n_epochs: int,
        embeddings: bool,
        embedding_dim: int,
        seq_length: int,
        units: list,
        model_name: str
    ):

    # get paths
    # home_path = os.path.dirname(os.getcwd())
    home_path = os.getcwd()
    data_path = home_path + '\\data\\'
    model_path = home_path + '\\models\\'
    # results_path = home_path + '\\a4\\results\\'
    
    # get text data
    fname = 'shakespeare.txt'
    fpath = data_path + fname
    
    # get text data
    data = getCharData(fpath)
    
    if embeddings:
        # train word2vec
        w2v_model = gsm.Word2Vec(
            sentences=data,
            vector_size=embedding_dim,
            window=20,
            min_count=1,
            workers=1
        )
        
        # extract word vectors
        char_vecs = w2v_model.wv
    
    # split data into words
    data = [char
              for sentence in data
                  for char in sentence
    ]
    
    # create word-key-word mapping
    keyToChar = dict(enumerate(np.unique(data)))
    charToKey = dict([(val, key) for key, val in keyToChar.items()])
    
    # define Y, w. one-hot encoded representations
    K = len(charToKey)
    Y = []
    for word in data:
        Y.append(oneHotEncode_v2(charToKey[word], K).astype('int8'))
    
    # compute sequences
    # seq_length = seq_length
    Y_seqs = []
    for i in range(len(Y) - seq_length):
        Y_seqs.append(Y[i:i+seq_length])
       
    if embeddings:
        
        keyToVec = dict([(key, char_vecs.get_vector(char)) for key, char in keyToChar.items()])
        charToVec = dict([(char, char_vecs.get_vector(char)) for _, char in keyToChar.items()])
        
        X = []
        for word in data:
            X.append(charToVec[word].astype('float64'))
            X_seqs = []
            for i in range(len(X) - seq_length):
                X_seqs.append(X[i:i+seq_length])
        
        X_seqs = X_seqs[:-1]
        Y_seqs = Y_seqs[1:]

    else:
        X_seqs, Y_seqs = Y_seqs[:-1], Y_seqs[1:]
    
    # get val and test data
    train_frac = 0.9
    train_n = int(len(X_seqs) * train_frac)
    val_n = len(X_seqs) - train_n
    
    X_train, Y_train = X_seqs[:train_n], Y_seqs[:train_n]
    X_val, Y_val = X_seqs[train_n:], Y_seqs[train_n:]
    
    # define model params
    # m = 100
    K_out  = len(keyToChar)
    K_in = embeddings * embedding_dim + (1 - embeddings) * K_out
    sigma = 0.1
    
    recurrentNet = LSTM(
        K_in=K_in,
        K_out=K_out,
        units=units,
        sigma=sigma,
        optimizer='adagrad',
        embeddings=embeddings,
        seed=2)
    
    if embeddings:
        recurrentNet.keyToVec = keyToVec
    
    # save best weights
    weights_best = recurrentNet.layers.copy()
    
    epoch_n = 0
    print ('\n------EPOCH {}--------\n'.format(epoch_n))
    
    trainLossHist = []
    trainLoss_smooth, _ = recurrentNet.computeCost(
        np.vstack(X_train[0]), 
        np.vstack(Y_train[0])
    )
    
    valLossHist = []
    valLoss_smooth, _ = recurrentNet.computeCost(
        np.vstack(X_val[0]), 
        np.vstack(Y_val[0])
    )
    valLoss_best = valLoss_smooth
    
    
    n = len(X_train)
    e = 0
    for i in range(2000000):
        # train net
        x, y = np.vstack(X_train[e]), np.vstack(Y_train[e])
        recurrentNet.train(
            x, 
            y, 
            eta=0.1, 
            t = 1
        )
        
        # get smoothed training loss
        trainLoss, _ = recurrentNet.computeCost(x, y)
        trainLoss_smooth = 0.999 * trainLoss_smooth + 0.001 * trainLoss
        
        # get validation loss w. random validation sample
        randIdx = np.random.randint(val_n)
        xVal, yVal = np.vstack(X_val[randIdx]), np.vstack(Y_val[randIdx])
        valLoss, _ = recurrentNet.computeCost(xVal, yVal)
        valLoss_smooth = 0.999 * valLoss_smooth + 0.001 * valLoss
    
        if valLoss_smooth < valLoss_best:
            weights_best = recurrentNet.layers.copy()
            valLoss_best = valLoss_smooth
    
        if (i % 100 == 0) and i > 0:
            trainLossHist.append(trainLoss_smooth)
            valLossHist.append(valLoss_smooth)
            print('Iteration {}, Train LOSS: {}, Val LOSS: {}'.format(
                i, 
                trainLoss_smooth,
                valLoss_smooth
            ))
            
        if i % 1000 == 0:
            sequence = recurrentNet.synthesizeText(
                x0=x[:1], 
                n=250
            )
            
            # convert to chars and print sequence
            sequence = ''.join([keyToChar[key] for key in sequence])
            print('\nGenerated sequence \n\n {}\n'.format(sequence))
            
        # update e
        if e < (n - seq_length):
            e += seq_length
        else:
            epoch_n += 1
            print ('\n------EPOCH {}--------\n'.format(epoch_n))
            
            with open(model_path + '{}_e{}'.format(model_name, epoch_n), 'wb') as fo:
                pickle.dump(weights_best, fo)
            
            if epoch_n == n_epochs:
                break
            
            e = 0
            
    return trainLossHist, valLossHist


import os
import json
from datetime import datetime
import numpy as np
from misc import oneHotEncode, readData, prepareData, generateSequences
from model import VanillaRNN, LSTM, LSTM_2L
from plot_methods import *
import random

# get paths
home_path = os.getcwd()
data_path = home_path + '/data/'
plot_path = home_path + '/plots/'
log_path = home_path + '/logs/'


# ========================================================================== #
# ------------------------ DD2424 PROJECT - LSTMs -------------------------- #
#--------------------------------------------------------------------------- #
# --- By: Oskar Stigland, Daniela Eklund, Daniel Hartler, August Tengland--- #
# ========================================================================== #


def main():

    # define globally for access in helper functions
    global keyToChar
    global charToKey
    global seq_length

    # define params
    m = 100  # dimensionality of its hidden state
    sigma = 0.1
    seq_length = 25

    print("Processing Data...")

    # get text data
    fname = 'shakespeare.txt'
    fpath = data_path + fname

    # read text file
    data = readData(fpath)

    # create word-key-word mapping
    keyToChar, charToKey = prepareData(data)
    
    K = len(keyToChar)  # nr of unique characters

    # define X, w. one-hot encoded representations of sequences
    data = oneHotEncode(np.array([charToKey[char] for char in data]))
    X_train, X_val = generateSequences(data, seq_length)

    """ 
    TODO: 
    Implement batches, which are composed of multiple sequences
    After this we only shuffle batch order, not the sequence order inside of them
    """

    # Shuffle sequences in X
    # random.seed(42)
    # random.shuffle(X)


    # init networks, replace class name to instantiate differnent models
    # Available RNN-models: ['VanillaRNN', 'LSTM', 'LSTM_2L']
    
    vrnn = VanillaRNN(
        K=K,
        m=m,
        sigma=sigma,
        seed=2
    )

    lstm = LSTM(
        K=K,
        m=m,
        sigma=sigma,
        seed=2
    )


    lstm_2l = LSTM_2L(
        K=K,
        m=m,
        sigma=sigma,
        seed=2
    )

    num_iterations = 100

    vrnn, lossHistVrnn_train, lossHistVrnn_test = runTraining(vrnn, X_train, X_val, num_iterations)
    #generateAndLogSequence(vrnn, X_train, num_iterations, lossHistVrnn_train[-1])
    #plotLoss(vrnn, lossHistVrnn_train, num_iterations, 'train')
    #plotLoss(vrnn, lossHistVrnn_test, num_iterations, 'validation')

    lstm, lossHistLstm_train, lossHistLstm_test = runTraining(lstm, X_train, X_val, num_iterations)
    #generateAndLogSequence(lstm, X, num_iterations, lossHistLstm[-1])
    #plotLoss(lstm, lossHistLstm_train, num_iterations, 'train')
    #plotLoss(lstm, lossHistLstm_test, num_iterations, 'validation')

    lstm_2l, lossHistLstm2_train, lossHistLstm2_test = runTraining(lstm_2l, X_train, X_val, num_iterations)
    #generateAndLogSequence(lstm_2l, X, num_iterations, lossHistLstm2[-1])
    #plotLoss(lstm, lossHistLstm2_train, num_iterations, 'train')
    #plotLoss(lstm, lossHistLstm2_test, num_iterations, 'validation')

    # rnn_list = [vrnn,lstm,lstm_2l]
    # lossHist_list_train = [lossHistVrnn_train, lossHistLstm_train, lossHistLstm2_train]
    # lossHist_list_test = [lossHistVrnn_test, lossHistLstm_test, lossHistLstm2_test]
    # multiPlotLoss(rnn_list, num_iterations, lossHist_list_train, 'Training')
    # multiPlotLoss(rnn_list, num_iterations, lossHist_list_test, 'Validation')

    # runEtaSigmaGridSearch(X_train, X_val,K,m,num_iterations)

    runHiddenLayerSearch(X_train, X_val,K,m,sigma,num_iterations)


# =====================================================
# ----------------- HELPER METHODS --------------------
# =====================================================

def runHiddenLayerSearch(X_train, X_val,K,m,sigma,num_iterations): 

    m_list = [10, 50, 100, 150, 200]

    lossHistLstm_list_train = []
    lossHistLstm_list_val = []

    for m in m_list:
        print("Testing new params:")
        print("m = " + str(m))

        lstm = LSTM(
            K=K,
            m=m,
            sigma=sigma,
            seed=2
        )


        lstm, lossHistLstm_train, lossHistLstm_test = runTraining(lstm, X_train, X_val, num_iterations)

        lossHistLstm_list_train.append(lossHistLstm_train)
        lossHistLstm_list_val.append(lossHistLstm_test)

    multiPlotLossHiddenLayer('LSTM', num_iterations, lossHistLstm_list_train, 'Training')
    multiPlotLossHiddenLayer('LSTM', num_iterations, lossHistLstm_list_val, 'Validation')




def runEtaSigmaGridSearch(X_train, X_val,K,m,num_iterations): 

    etas = [0.01, 0.05, 0.1, 0.2, 0.5]
    sigmas = [0.5, 0.2, 0.1, 0.05, 0.01]

    lossHissList_lstm  = np.zeros((len(etas),len(sigmas))) 
    lossHissList_lstm2 = np.zeros((len(etas),len(sigmas))) 

    for r,eta in enumerate(etas):
       for c,sigma in enumerate(sigmas):
            print("Testing new params:")
            print("eta = " + str(eta) + ", sigma = " + str(sigma))

            lstm = LSTM(
                K=K,
                m=m,
                sigma=sigma,
                seed=2
            )

            lstm_2l = LSTM_2L(
                K=K,
                m=m,
                sigma=sigma,
                seed=2
            )

            lstm, lossHistLstm = runTraining(lstm, X_train, X_val, num_iterations, eta)
            lstm_bestloss = min(lossHistLstm)
            lossHissList_lstm[r][c] = lstm_bestloss

            lstm_2l, lossHistLstm2 = runTraining(lstm_2l, X_train, X_val, num_iterations, eta)
            lstm2_bestloss = min(lossHistLstm2)
            lossHissList_lstm2[r][c] = lstm2_bestloss

    paramSearchHeatmap(lstm, num_iterations, 'eta', etas, 'sigma', sigmas, lossHissList_lstm)  
    paramSearchHeatmap(lstm_2l, num_iterations, 'eta', etas, 'sigma', sigmas, lossHissList_lstm2)  

def runTraining(rnn, X_train, X_val, num_iterations, eta=0.1):

    m = rnn.m

    # save best weights
    weights_best = rnn.weights.copy()

    epoch_n = 0
    print('\n------EPOCH {}--------\n'.format(epoch_n))

    lossHist_train = []
    lossHist_val = []
    loss_smooth_train = rnn.computeLoss(X_train[0][:-1], X_train[0][1:])
    loss_smooth_val = rnn.computeLoss(X_val[0][:-1], X_val[0][1:])
    loss_best_train = loss_smooth_train
    loss_best_val = loss_smooth_val
    lossHist_train.append(loss_smooth_train)
    lossHist_val.append(loss_smooth_val)
    print('Iteration 0, TRAIN LOSS: {}, VAL LOSS: {}'.format(loss_smooth_train, loss_smooth_val))

    n = len(X_train)
    e = 0
    for i in range(1, num_iterations+1):

        x_seq = X_train[e][:-1]
        y_seq = X_train[e][1:]

        x_val = X_val[e % len(X_val)][:-1]
        y_val = X_val[e % len(X_val)][1:]

        rnn.train(x_seq, y_seq, eta=eta)
        loss_train = rnn.computeLoss(x_seq, y_seq)
        loss_val = rnn.computeLoss(x_val, y_val)

        loss_smooth_train = 0.999 * loss_smooth_train + 0.001 * loss_train
        loss_smooth_val = 0.999 * loss_smooth_val + 0.001 * loss_val

        if loss_smooth_train < loss_best_train:
            weights_best = rnn.weights.copy()
            loss_best_train = loss_smooth_train

        if loss_smooth_val < loss_best_val:
            weights_best = rnn.weights.copy()
            loss_best_val = loss_smooth_val

        if (i % 10 == 0) and i > 0:
            lossHist_train.append(loss_smooth_train)
            lossHist_val.append(loss_smooth_val)

            if i % 100 == 0:
                print('Iteration {}, TRAIN LOSS: {}, VAL LOSS: {}'.format(i, loss_smooth_train, loss_smooth_val))

        if i % 1000 == 0:

            sequence = rnn.synthesizeText(
                x0=X_train[e][:1],
                n=250
            )

            # convert to chars and print sequence
            sequence = ''.join([keyToChar[key] for key in sequence])
            print('\nGenerated sequence \n\n {}\n'.format(sequence))

        # update e
        if e < (n - seq_length):
            e += seq_length
        else:
            e = 0

            rnn.hprev = np.zeros(shape=(m, 1))

            epoch_n += 1
            print('\n------EPOCH {}--------\n'.format(epoch_n))

            if epoch_n >= 4:
                break

    rnn.weights = weights_best
    logTrainingResults(rnn, num_iterations, lossHist_train, loss_best_train, 'train')
    logTrainingResults(rnn, num_iterations, lossHist_val, loss_best_val, 'val')
    logLossHiss(rnn, num_iterations, lossHist_train, 'train')
    logLossHiss(rnn, num_iterations, lossHist_val, 'val')
    return rnn, lossHist_train, lossHist_val

def logTrainingResults(rnn, num_iterations, lossHist, loss_best, result_type):
    now = datetime.now()
    f = open(log_path + "trainingResults.txt", "a")
    f.write("New Training Log, time: " + now.strftime("%d/%m/%Y %H:%M:%S") + "\n")
    f.write("Model type: " + rnn.type + ", m: " + str(rnn.m) + ", sigma: " + str(rnn.sigma) + "\n")
    f.write("Num_iterations: " + str(num_iterations) + ", current_loss: " + str(lossHist[-1]) + ", best_loss: " + str(loss_best) + ", result_type: " + str(result_type) + "\n")
    f.write("\n")
    f.close()

def logLossHiss(rnn, num_iterations, lossHist, result_type): 
    now = datetime.now()
    f = open(log_path + 'loss_hiss/' + str(rnn.type) + '_' + str(num_iterations) + '_' + str(result_type) + '_' + now.strftime("%H%M%S") + '.txt', "w")
    f.writelines(str(lossHist))
    f.write("\n")
    f.close()
    
def generateAndLogSequence(rnn, X, num_iterations, loss_smooth):
    sequence = printAndReturnSequence(rnn, X)
    now = datetime.now()
    f = open(log_path + '/text_generation/' + str(rnn.type) + '_' + str(num_iterations) + '_text_generation.txt', "w")
    f.write("Text generation for model type " + rnn.type + " at time: " + now.strftime("%d/%m/%Y %H:%M:%S") + "\n")
    f.write("Num_iterations: " + str(num_iterations) + ", current_loss: " + str(loss_smooth) + "\n")
    f.write("Generated text: \n \n")
    f.write(sequence)
    f.write("\n")
    f.close()

def printAndReturnSequence(rnn, X):

    sequence = rnn.synthesizeText(
            x0=X[0][:1],
            n=1000
        )

    # convert to chars and print sequence
    sequence = ''.join([keyToChar[key] for key in sequence])
    print('\nGenerated sequence \n\t {}\n'.format(sequence))

    return sequence


# =====================================================
# -------------------- MAIN CALL ----------------------
# =====================================================


if __name__ == "__main__":
    main()
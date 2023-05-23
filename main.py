
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
    X = generateSequences(data, seq_length)

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

    # vrnn, lossHistVrnn = runTraining(vrnn, X, num_iterations)
    # generateAndLogSequence(vrnn, X, num_iterations, lossHistVrnn[-1])
    # plotLoss(vrnn, lossHistVrnn, num_iterations)

    # lstm, lossHistLstm = runTraining(lstm, X, num_iterations)
    # generateAndLogSequence(lstm, X, num_iterations, lossHistLstm[-1])
    # plotLoss(lstm, lossHistLstm, num_iterations)

    # lstm_2l, lossHistLstm2 = runTraining(lstm_2l, X, num_iterations)
    # generateAndLogSequence(lstm_2l, X, num_iterations, lossHistLstm2[-1])
    # plotLoss(lstm_2l, lossHistLstm2, num_iterations)

    # rnn_list = [vrnn,lstm,lstm_2l]
    # lossHist_list = [lossHistVrnn, lossHistLstm, lossHistLstm2]
    # multiPlotLoss(rnn_list, num_iterations, lossHist_list)

    # runEtaSigmaGridSearch(X,K,m,num_iterations)

    runHiddenLayerSearch(X,K,m,sigma,num_iterations)


# =====================================================
# ----------------- HELPER METHODS --------------------
# =====================================================

def runHiddenLayerSearch(X,K,m,sigma,num_iterations): 

    m_list = [10, 50, 100, 150, 200]

    lossHistVrnn_list = []
    lossHistLstm_list = []
    lossHistLstm2_list = []

    for m in m_list:
        print("Testing new params:")
        print("m = " + str(m))

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

        vrnn, lossHistVrnn = runTraining(vrnn, X, num_iterations)
        lstm, lossHistLstm = runTraining(lstm, X, num_iterations)
        lstm_2l, lossHistLstm2 = runTraining(lstm_2l, X, num_iterations)

        lossHistVrnn_list.append(lossHistVrnn)
        lossHistLstm_list.append(lossHistLstm)
        lossHistLstm2_list.append(lossHistLstm2)

    multiPlotLossHiddenLayer('VanillaRNN', num_iterations, lossHistVrnn_list)
    multiPlotLossHiddenLayer('LSTM', num_iterations, lossHistLstm_list)
    multiPlotLossHiddenLayer('LSTM_L2', num_iterations, lossHistLstm2_list)




def runEtaSigmaGridSearch(X,K,m,num_iterations): 

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

            lstm, lossHistLstm = runTraining(lstm, X, num_iterations, eta)
            lstm_bestloss = min(lossHistLstm)
            lossHissList_lstm[r][c] = lstm_bestloss

            lstm_2l, lossHistLstm2 = runTraining(lstm_2l, X, num_iterations, eta)
            lstm2_bestloss = min(lossHistLstm2)
            lossHissList_lstm2[r][c] = lstm2_bestloss

    paramSearchHeatmap(lstm, num_iterations, 'eta', etas, 'sigma', sigmas, lossHissList_lstm)  
    paramSearchHeatmap(lstm_2l, num_iterations, 'eta', etas, 'sigma', sigmas, lossHissList_lstm2)  

def runTraining(rnn, X, num_iterations, eta=0.1):

    m = rnn.m

    # save best weights
    weights_best = rnn.weights.copy()

    epoch_n = 0
    print('\n------EPOCH {}--------\n'.format(epoch_n))

    lossHist = []
    loss_smooth = rnn.computeLoss(X[0][:-1], X[0][1:])
    loss_best = loss_smooth
    lossHist.append(loss_smooth)
    print('Iteration 0, LOSS: {}'.format(loss_smooth))

    n = len(X)
    e = 0
    for i in range(1, num_iterations+1):

        x_seq = X[e][:-1]
        y_seq = X[e][1:]

        rnn.train(x_seq, y_seq, eta=eta)
        loss = rnn.computeLoss(x_seq, y_seq)

        loss_smooth = 0.999 * loss_smooth + 0.001 * loss
        if loss_smooth < loss_best:
            weights_best = rnn.weights.copy()
            loss_best = loss_smooth

        if (i % 10 == 0) and i > 0:
            lossHist.append(loss_smooth)

            if i % 100 == 0:
                print('Iteration {}, LOSS: {}'.format(i, loss_smooth))

        if i % 1000 == 0:

            sequence = rnn.synthesizeText(
                x0=X[e][:1],
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
    logTrainingResults(rnn, num_iterations, lossHist, loss_best)
    logLossHiss(rnn, num_iterations, lossHist)
    return rnn, lossHist

def logTrainingResults(rnn, num_iterations, lossHist, loss_best):
    now = datetime.now()
    f = open(log_path + "trainingResults.txt", "a")
    f.write("New Training Log, time: " + now.strftime("%d/%m/%Y %H:%M:%S") + "\n")
    f.write("Model type: " + rnn.type + ", m: " + str(rnn.m) + ", sigma: " + str(rnn.sigma) + "\n")
    f.write("Num_iterations: " + str(num_iterations) + ", current_loss: " + str(lossHist[-1]) + ", best_loss: " + str(loss_best) + "\n")
    f.write("\n")
    f.close()

def logLossHiss(rnn, num_iterations, lossHist): 
    now = datetime.now()
    f = open(log_path + 'loss_hiss/' + str(rnn.type) + '_' + str(num_iterations) + '_' + now.strftime("%H%M%S") + '.txt', "w")
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
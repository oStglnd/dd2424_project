
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from misc import oneHotEncode, readData, prepareData, generateSequences
from model import VanillaRNN, LSTM, LSTM_2L
import random

# get paths
home_path = os.getcwd()
data_path = home_path + '/data/'
plot_path = home_path + '/plots/'


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
    m = 200  # dimensionality of its hidden state
    sigma = 0.01
    seq_length = 25


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
    random.seed(42)
    random.shuffle(X)


    # init networks, replace class name to instantiate differnent models
    # Available RNN-models: ['VanillaRNN', 'LSTM', 'LSTM_2L']
    """ 
    TODO: 
    Create LSTM_2L (2-layer LSTM)
    """
    rnn = VanillaRNN(
        K=K,
        m=m,
        sigma=sigma,
        seed=2
    )


    num_iterations = 1000
    rnn, lossHist = runTraining(rnn, X, num_iterations)
    plotLoss(lossHist)
    printSequence(rnn, X)


# =====================================================
# ----------------- HELPER METHODS --------------------
# =====================================================


def runTraining(rnn, X, num_iterations):

    m = rnn.m

    # save best weights
    weights_best = rnn.weights.copy()

    epoch_n = 0
    print('\n------EPOCH {}--------\n'.format(epoch_n))

    lossHist = []
    loss_smooth = rnn.computeLoss(X[0][:-1], X[0][1:])
    loss_best = loss_smooth

    n = len(X)
    e = 0
    for i in range(num_iterations):

        x_seq = X[e][:-1]
        y_seq = X[e][1:]

        rnn.train(x_seq, y_seq, eta=0.1)
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
        if e < (n - seq_length - 1):
            e += seq_length + 1
        else:
            e = 0

            rnn.hprev = np.zeros(shape=(m, 1))

            epoch_n += 1
            print('\n------EPOCH {}--------\n'.format(epoch_n))

            if epoch_n >= 4:
                break

    rnn.weights = weights_best
    
    return rnn, lossHist  


def plotLoss(lossHist):
    # plot results
    steps = [step * 10 for step in range(len(lossHist))]
    plt.plot(steps, lossHist, 'r', linewidth=1.5, alpha=1.0, label='Loss')
    plt.xlim(0, steps[-1])
    plt.xlabel('Step')
    plt.ylabel('', rotation=0, labelpad=20)
    plt.title('Smooth loss for $4$ epochs')
    plt.savefig(plot_path + 'rnn_loss.png', dpi=200)
    plt.show()
    

def printSequence(rnn, X):

    sequence = rnn.synthesizeText(
            x0=X[0][:1],
            n=200
        )

    # convert to chars and print sequence
    sequence = ''.join([keyToChar[key] for key in sequence])
    print('\nGenerated sequence \n\t {}\n'.format(sequence))



# =====================================================
# -------------------- MAIN CALL ----------------------
# =====================================================


if __name__ == "__main__":
    main()
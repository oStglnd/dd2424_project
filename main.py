
import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from misc import oneHotEncode, readData, prepareData, generateSequences, cleanData, generateTrainingData
from model import VanillaRNN, LSTM, LSTM_2L
import random
from sklearn.utils import shuffle

# get paths
home_path = os.getcwd()
data_path = home_path + '/data/'
plot_path = home_path + '/plots/'
log_path = home_path + '/logs/'


# ========================================================================== #
# ------------------------ DD2424 PROJECT - LSTMs -------------------------- #
# --------------------------------------------------------------------------- #
# --- By: Oskar Stigland, Daniela Eklund, Daniel Hartler, August Tengland--- #
# ========================================================================== #


def main():

    # define globally for access in helper functions
    global keyToToken
    global tokenToKey
    global seq_length
    global K

    # define params
    m = 200  # dimensionality of its hidden state
    sigma = 0.01
    seq_length = 10

    print("Processing Data...")

    # get text data
    fname = 'shakespeare.txt'
    fpath = data_path + fname

    # read text file
    data = readData(fpath)

    # clean data
    tokens = cleanData(data, stopwords=True)

    # create word-key-word mapping
    keyToToken, tokenToKey = prepareData(tokens)
    K = len(keyToToken)  # nr of unique characters

    # define X, w. one-hot encoded representations of sequences
    # data = oneHotEncode(np.array([charToKey[char] for char in data]))
    X, y = generateSequences(tokens, seq_length)

    # Shuffle sequences in X
    random.seed(42)
    X_shuffled, y_shuffled = shuffle(X, y)

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

    num_iterations = 200000

    # vrnn, lossHist = runTraining(vrnn, X_shuffled, y_shuffled, num_iterations)
    # generateAndLogSequence(vrnn, X_shuffled, num_iterations, lossHist[-1])
    # plotLoss(vrnn, lossHist, num_iterations)

    lstm, lossHist = runTraining(lstm, X_shuffled, y_shuffled, num_iterations)
    generateAndLogSequence(lstm, X_shuffled, num_iterations, lossHist[-1])
    plotLoss(lstm, lossHist, num_iterations)

    # lstm_2l, lossHist = runTraining(lstm_2l, X_shuffled, y_shuffled, num_iterations)
    # generateAndLogSequence(lstm_2l, X_shuffled, num_iterations, lossHist[-1])
    # plotLoss(lstm_2l, lossHist, num_iterations)


# =====================================================
# ----------------- HELPER METHODS --------------------
# =====================================================


def runTraining(rnn, X, y, num_iterations):

    m = rnn.m

    # save best weights
    weights_best = rnn.weights.copy()

    epoch_n = 0
    print('\n------EPOCH {}--------\n'.format(epoch_n))

    lossHist = []
    loss_smooth = rnn.computeLoss(oneHotEncode(np.array([tokenToKey[token] for token in X[0]]), K), oneHotEncode(
        np.array([tokenToKey[token] for token in y[0]]), K))
    loss_best = loss_smooth
    lossHist.append(loss_smooth)
    print('Iteration 0, LOSS: {}'.format(loss_smooth))

    n_sequences = len(X)
    e = 0
    for i in range(1, num_iterations+1):

        x_seq = oneHotEncode(
            np.array([tokenToKey[token] for token in X[e]]), K)
        y_seq = oneHotEncode(
            np.array([tokenToKey[token] for token in y[e]]), K)

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
                x0=x_seq[:1],
                n=10
            )

            # convert to words and print sequence
            _ = printSequence(sequence)

        # update e
        if e < (n_sequences - 1):
            e += 1
        else:
            e = 0

            rnn.hprev = np.zeros(shape=(m, 1))

            epoch_n += 1
            print('\n------EPOCH {}--------\n'.format(epoch_n))

            if epoch_n >= 4:
                break

    rnn.weights = weights_best
    logTrainingResults(rnn, num_iterations, loss_smooth, loss_best)

    return rnn, lossHist


def logTrainingResults(rnn, num_iterations, loss_smooth, loss_best):
    now = datetime.now()
    f = open(log_path + "trainingResults.txt", "a")
    f.write("New Training Log, time: " +
            now.strftime("%d/%m/%Y %H:%M:%S") + "\n")
    f.write("Model type: " + rnn.type + ", num_iterations: " + str(num_iterations) +
            ", current_loss: " + str(loss_smooth) + ", best_loss: " + str(loss_best) + "\n")
    f.write("\n")
    f.close()


def generateAndLogSequence(rnn, X, num_iterations, loss_smooth):
    sequence = printAndReturnSequence(rnn, X)
    now = datetime.now()
    f = open(log_path + str(rnn.type) + '_' +
             str(num_iterations) + '_text_generation.txt', "w")
    f.write("Text generation for model type " + rnn.type +
            " at time: " + now.strftime("%d/%m/%Y %H:%M:%S") + "\n")
    f.write("Num_iterations: " + str(num_iterations) +
            ", current_loss: " + str(loss_smooth) + "\n")
    f.write("Generated text: \n \n")
    f.write(sequence)
    f.write("\n")
    f.close()


def plotLoss(rnn, lossHist, num_iterations):

    plotTitle = "Smooth loss for {rnnType}, after {num_iterations} iterations".format(
        rnnType=rnn.type, num_iterations=num_iterations)

    # plot results
    steps = [step * 10 for step in range(len(lossHist))]
    plt.plot(steps, lossHist, 'r', linewidth=1.5, alpha=1.0, label='Loss')
    plt.xlim(0, steps[-1])
    plt.xlabel('Iterations')
    plt.ylabel('Loss', rotation=0, labelpad=20)
    plt.title(plotTitle)
    plt.savefig(plot_path + str(rnn.type) + '_' +
                str(num_iterations) + '_iter_loss.png', dpi=200)
    plt.clf()


def printSequence(seq, i=0):
    sequence = ' '.join([keyToToken[key] for key in seq])
    sequence += '.'
    sequence = sequence.capitalize()

    if i == 0:
        print('\nGenerated sequence: \n\n{}'.format(sequence))

    else:
        print('{}'.format(sequence))

    return sequence


def printAndReturnSequence(rnn, X, n_seq=10):
    sequence_str = ''
    for i in range(n_seq):
        sequence = rnn.synthesizeText(
            x0=oneHotEncode(np.array([tokenToKey[token]
                            for token in X[i]]), K)[:1],
            n=10
        )

        # convert to chars and print sequence
        sequence_str += printSequence(sequence, i)
    # sequence = ''.join([keyToToken[key] for key in sequence])
    # print('\nGenerated sequence \n\t {}\n'.format(sequence))

    return sequence_str


# =====================================================
# -------------------- MAIN CALL ----------------------
# =====================================================


if __name__ == "__main__":
    main()

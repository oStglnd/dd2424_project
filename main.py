
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from misc import oneHotEncode, readData, prepareData, generateSequences
from model import recurrentNeuralNetwork, LSTM
import random

# get paths
home_path = os.getcwd()
data_path = home_path + '/data/'
plot_path = home_path + '/plots/'

# get text data
fname = 'shakespeare.txt'
fpath = data_path + fname

# read text file
data = readData(fpath)

# create word-key-word mapping
keyToChar, charToKey = prepareData(data)

# define params
K = len(keyToChar)  # nr of unique characters
m = 200  # dimensionality of its hidden state
sigma = 0.01
seq_length = 25

# define X, w. one-hot encoded representations of sequences
data = oneHotEncode(np.array([charToKey[char] for char in data]))
X = generateSequences(data, seq_length)

# Shuffle sequences in X
random.seed(42)
random.shuffle(X)

# init networks
lstmNet = LSTM(
    K=K,
    m=m,
    sigma=sigma,
    seed=2
)

# lstmNet = recurrentNeuralNetwork(
#     K=K,
#     m=m,
#     sigma=sigma,
#     seed=2
# )

# save best weights
weights_best = lstmNet.weights.copy()

epoch_n = 0
print('\n------EPOCH {}--------\n'.format(epoch_n))

lossHist = []
loss_smooth = lstmNet.computeLoss(X[0][:-1], X[0][1:])
loss_best = loss_smooth

n = len(X)
e = 0
for i in range(2000000):
    x_seq = X[e][:-1]
    y_seq = X[e][1:]

    lstmNet.train(x_seq, y_seq, eta=0.1)
    loss = lstmNet.computeLoss(x_seq, y_seq)

    loss_smooth = 0.999 * loss_smooth + 0.001 * loss
    if loss_smooth < loss_best:
        weights_best = lstmNet.weights.copy()
        loss_best = loss_smooth

    if (i % 10 == 0) and i > 0:
        lossHist.append(loss_smooth)

        if i % 100 == 0:
            print('Iteration {}, LOSS: {}'.format(i, loss_smooth))

    if i % 1000 == 0:
        sequence = lstmNet.synthesizeText(
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
        lstmNet.hprev = np.zeros(shape=(m, 1))

        epoch_n += 1
        print('\n------EPOCH {}--------\n'.format(epoch_n))

        if epoch_n >= 4:
            break

# # plot results
# steps = [step * 10 for step in range(len(lossHist))]
# plt.plot(steps, lossHist, 'r', linewidth=1.5, alpha=1.0, label='Loss')
# plt.xlim(0, steps[-1])
# plt.xlabel('Step')
# plt.ylabel('', rotation=0, labelpad=20)
# plt.title('Smooth loss for $4$ epochs')
# # plt.legend(loc='upper right')
# plt.savefig(plot_path + 'rnn_loss.png', dpi=200)
# plt.show()

# recurrentNet.weights = weights_best
sequence = lstmNet.synthesizeText(
    x0=X[0][:1],
    n=200
)

# convert to chars and print sequence
sequence = ''.join([keyToChar[key] for key in sequence])
print('\nGenerated sequence \n\t {}\n'.format(sequence))

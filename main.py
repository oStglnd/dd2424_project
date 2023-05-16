
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from misc import oneHotEncode
from model import recurrentNeuralNetwork, LSTM

# get paths
# home_path = os.path.dirname(os.getcwd())
home_path = os.getcwd()
data_path = home_path + '\\data\\'
plot_path = home_path + '\\plots\\'
# results_path = home_path + '\\a4\\results\\'

# get text data
fname = 'shakespeare.txt'
fpath = data_path + fname

# read text file
with open(fpath, 'r') as fo:
    data = fo.readlines()
    
# split lines into words and words into chars
data = [char 
            for line in data
                for word in list(line)
                    for char in list(word)]

# create word-key-word mapping
keyToChar = dict(enumerate(np.unique(data)))
charToKey = dict([(val, key) for key, val in keyToChar.items()])

# define params
K  = len(keyToChar)
m = 100
sigma = 0.01
seq_length = 20

# define X, w. one-hot encoded representations
data = oneHotEncode(np.array([charToKey[char] for char in data]))
X = []
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])

# init networks
lstmNet = LSTM(
    K=K,
    m=m,
    sigma=sigma,
    seed=2
)

# save best weights
weights_best = lstmNet.weights.copy()

epoch_n = 0
print ('\n------EPOCH {}--------\n'.format(epoch_n))

lossHist = []
loss_smooth = lstmNet.computeLoss(X[0], X[1])
loss_best = loss_smooth

n = len(X)
e = 0
for i in range(2000000):
    lstmNet.train(X[e], X[e+1], eta=0.1)
    loss = lstmNet.computeLoss(X[e], X[e+1])
    
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
            x0=X[e+1][:1], 
            n=250
        )
        
        # convert to chars and print sequence
        sequence = ''.join([keyToChar[key] for key in sequence])
        print('\nGenerated sequence \n\t {}\n'.format(sequence))
        
    # update e
    if e < (n - seq_length):
        e += seq_length
    else:
        e = 0
        lstmNet.hprev = np.zeros(shape=(m, 1))
        
        epoch_n += 1
        print ('\n------EPOCH {}--------\n'.format(epoch_n))
        
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
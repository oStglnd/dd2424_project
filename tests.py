
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from misc import oneHotEncode
from model import VanillaRNN, LSTM, LSTM_2L

# get paths
# home_path = os.path.dirname(os.getcwd())
home_path = os.getcwd()
data_path = home_path + '\\data\\'
plot_path = home_path + '\\plots\\'
# results_path = home_path + '\\a4\\results\\'

# get text data
fname = 'shakespeare.txt'
fpath = data_path + fname

print("Processing Data...")

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
sigma = 0.1
seq_length = 2

# define X, and Y, w. one-hot encoded representations
data = oneHotEncode(np.array([charToKey[char] for char in data]))
X = []
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])

# init networks
rnn = LSTM_2L(
    K=K,
    m=m,
    sigma=sigma,
    seed=2
)

gradsListNum = rnn.computeGradsNumerical(
    X[1], 
    X[2],
    eps=1e-4
)

gradsList = rnn.computeGrads(
    X[1], 
    X[2],
)

print('\nGradient check:')

if rnn.type == "LSTM_2L": 
    for idx, subgradList in enumerate(gradsList):
        print("Checking gradient layer: " + str(idx+1) + ":")
        for key, grads in subgradList.items():
            print(np.min(np.abs(grads[:50, :50])))
            gradDiff = np.abs(grads[:50, :50]-gradsListNum[idx][key][:50, :50])
            gradDenom = np.maximum(1e-9, np.abs(grads[:50, :50])+np.abs(gradsListNum[idx][key][:50, :50]))
            W_gradDiffMax = np.max(gradDiff/gradDenom)
            print('\t max|{} - {}_num| = {:.10f}'.format(key, key, W_gradDiffMax))          

else: 
    for key, grads in gradsList.items():
        print(np.min(np.abs(grads[:50, :50])))
        gradDiff = np.abs(grads[:50, :50]-gradsListNum[key][:50, :50])
        gradDenom = np.maximum(1e-9, np.abs(grads[:50, :50])+np.abs(gradsListNum[key][:50, :50]))
        W_gradDiffMax = np.max(gradDiff/gradDenom)
        print('\t max|{} - {}_num| = {:.10f}'.format(key, key, W_gradDiffMax))


# lossHist = []
# smooth_loss, _ = recurrentNet.computeCost(X[0], X[1], lambd=0)

# n = len(X)
# e = 0
# for i in range(30000):
#     recurrentNet.train(X[e], X[e+1], lambd=0, eta=0.1)
#     loss, _ = recurrentNet.computeCost(X[e], X[e+1], lambd=0)
#     smooth_loss = 0.999 * smooth_loss + 0.001 * loss

#     if (i % 100 == 0) and i > 0:
#         lossHist.append(smooth_loss)
#         print('Iteration {}, LOSS: {}'.format(i, smooth_loss))
        
#     # if i % 1000 == 0:
#     #     sequence = recurrentNet.synthesizeText(
#     #         x0=X[e+1][:1], 
#     #         n=250
#     #     )
        
#     #     # convert to chars and print sequence
#     #     sequence = ''.join([keyToChar[key] for key in sequence])
#     #     print('\nGenerated sequence \n\t {}\n'.format(sequence))
        
#     # update e
#     if e < (n - seq_length):
#         e += seq_length
#     else:
#         e = 0
#         recurrentNet.hprev = np.zeros(shape=(m, 1))
        
# # plot results
# steps = [step * 100 for step in range(len(lossHist))]
# plt.plot(steps, lossHist, 'r', linewidth=1.5, alpha=1.0, label='Loss')
# plt.xlim(0, steps[-1])
# plt.xlabel('Step')
# plt.ylabel('', rotation=0, labelpad=20)
# plt.title('Smooth loss for small subset')
# # plt.legend(loc='upper right')
# plt.savefig(plot_path + 'grad_test_rnn.png', dpi=200)
# plt.show()
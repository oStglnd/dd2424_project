
import os
import numpy as np
import matplotlib.pyplot as plt

def softMax(S: np.array) -> np.array:
    """
    Parameters
    ----------
    S : dxN score matrix

    Returns
    -------
    S : dxN score matrix w. applied softmax activation
    """
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

class recurrentNeuralNetwork:
    def __init__(
            self, 
            K: int, 
            m: list,
            sigma: float,
            seed: int
        ):
        
        # init seed
        np.random.seed(seed)
        
        # init weight dims
        self.K = K
        self.m = m
        
        # init weight dict
        self.weights = {}
        self.momentum = {}
        
        # init bias/shift weights
        self.weights['b'] = np.zeros(shape=(self.m, 1))
        self.weights['c'] = np.zeros(shape=(self.K, 1))
        
        # init weight matrices
        self.weights['U'] = np.random.randn(self.m, self.K) * sigma
        self.weights['W'] = np.random.randn(self.m, self.m) * sigma
        self.weights['V'] = np.random.randn(self.K, self.m) * sigma
        
        for key, weight in self.weights.items():
            self.momentum[key] = np.zeros(shape=weight.shape)
        
        # initialize hprev
        self.hprev = np.zeros(shape=(self.m, 1))

    def synthesizeText(
            self,
            x0: np.array,
            n: int
        ) -> list:
        """
        Parameters
        ----------
        X0 : 1xK initial one-hot encodede entry
        n : length of sequence

        Returns
        -------
        xList : List of generated character indices
        """
        h = self.hprev
        xList = [x0]
        for _ in range(n):
            a = self.weights['W'] @ h + self.weights['U'] @ xList[-1].T + self.weights['b']
            h = np.tanh(a)
            o = self.weights['V'] @ h + self.weights['c']
            p = softMax(o)
            
            idxNext = np.random.choice(
                a=range(self.K), 
                p=np.squeeze(p)
            )
            
            x = np.zeros(shape=(1, self.K))
            x[0, idxNext] = 1
            xList.append(x)
        
        xList = [np.argmax(x) for x in xList]
        return xList
        

    def evaluate(
            self, 
            X: np.array,
            train: bool
        ) -> np.array:
        """
        Parameters
        ----------
        X : seq_len x K one-hot encoded sequence matrix
        Returns
        -------
        P : KxN score matrix w. softmax activation
        """
        probSeq = []        
        hList = [self.hprev.copy()]
        aList = []
        oList = []
        
        # iterate through recurrent block
        for x in X:
            a = self.weights['W'] @ hList[-1] + self.weights['U'] @ x[:, np.newaxis] + self.weights['b']
            h = np.tanh(a)
            o = self.weights['V'] @ h + self.weights['c']
            p = softMax(o)
            
            # save vals
            aList.append(a)
            hList.append(h)
            oList.append(o)
            probSeq.append(p)
        
        P = np.hstack(probSeq)
        A = np.hstack(aList)
        H = np.hstack(hList)
        O = np.hstack(oList)
        
        # update hprev
        if train:
            self.hprev = H[:, -1][:, np.newaxis]
        
        if train:
            # return P, aList, hList, oList
            return P, A, H, O
        else:
            return P
    
    def predict(
            self, 
            X: np.array
    ) -> np.array:
        """
        Parameters
        ----------
        X : seq_len x K one-hot encoded sequence matrix
        Returns
        -------
        P : KxN score matrix w. softmax activation
        """
        P = self.evaluate(X, train=False)
        preds = np.argmax(
            P, 
            axis=0
        )
        
        return preds
    
    def computeCost(
            self, 
            X: np.array, 
            Y: np.array
        ) -> float:
        """
        Parameters
        ----------
        X : seq_len x K one-hot encoded sequence matrix, at t
        Y : seq_len x K one-hot encoded sequence matrix, at t+1
        lambd : regularization parameter
        
        Returns
        -------
        l : cross entropy loss
        """
        # get probs
        P = self.evaluate(X, train=False)
        
        # evaluate loss term
        l = - np.sum(Y.T * np.log(P))
        
        return l
    
    def computeGrads(
            self, 
            X: np.array, 
            Y: np.array
        ) -> (np.array, np.array):
        """
        Parameters
        ----------
        X : seq_len x K one-hot encoded sequence matrix, at t
        Y : seq_len x K one-hot encoded sequence matrix, at t+1
        
        Returns
        -------
        grads : dictionary w. computed gradients
        """
        P, A, H, O = self.evaluate(X=X, train=True)
        g = -(Y.T - P)
        
        # get V grad
        V_grads = g @ H.T[1:]
        c_grads = np.sum(g, axis=1)[:, np.newaxis]
        
        # compute grads for a and h
        h_grad = g.T[-1] @ self.weights['V']
        a_grad = h_grad * (1 - np.square(np.tanh(A.T[-1])))
        
        # init lists for grads, a and h
        h_grads = [h_grad]
        a_grads = [a_grad]
        
        for g_t, a_t in zip(g.T[-2::-1], A.T[-2::-1]):
            
            h_grad = g_t @ self.weights['V'] + a_grad @ self.weights['W']
            a_grad = h_grad * (1 - np.square(np.tanh(a_t)))
        
            h_grads.append(h_grad)
            a_grads.append(a_grad)
        
        h_grads = np.vstack(h_grads[::-1]).T
        a_grads = np.vstack(a_grads[::-1]).T
        
        # get W grads
        W_grads = a_grads @ H.T[:-1]
        U_grads = a_grads @ X
        b_grads = np.sum(a_grads, axis=1)[:, np.newaxis]
        
        # save grads
        grads = {
            'V':V_grads,
            'W':W_grads,
            'U':U_grads,
            'b':b_grads,
            'c':c_grads
        }
        
        return grads
    
    def computeGradsNumerical(
            self, 
            X: np.array, 
            Y: np.array,
            eps: float,
        ) -> np.array:
        """
        Parameters
        ----------
        X : seq_len x K one-hot encoded sequence matrix, at t
        Y : seq_len x K one-hot encoded sequence matrix, at t+1
        eps: epsilon for incremental derivative calc.
        
        Returns
        -------
        W_gradsNum : numerically calculated gradients for weight martix (W)
        b_gradsNum : numerically calculated gradients for bias matrix (b)
        """

        # save initial weights
        gradsDict = {}

        for name, weight in self.weights.items():
            shape = weight.shape
            w_perturb = np.zeros(shape)
            w_gradsNum = np.zeros(shape)
            w_0 = weight.copy()
            
            for i in range(self.K):
                for j in range(min(shape[1], self.K)):
            # for i in range(shape[0]):
            #     for j in range(shape[1]):
            
                    # add perturbation
                    w_perturb[i, j] = eps
                    
                    # perturb weight vector negatively
                    # and compute cost
                    w_tmp = w_0 - w_perturb
                    self.weights[name] = w_tmp
                    _, cost1 = self.computeCost(X, Y)
                
                    # perturb weight vector positively
                    # and compute cost
                    w_tmp = w_0 + w_perturb
                    self.weights[name] = w_tmp
                    _, cost2 = self.computeCost(X, Y)
                    lossDiff = (cost2 - cost1) / (2 * eps)
                    
                    # get numerical grad f. W[i, j]
                    w_gradsNum[i, j] = lossDiff
                    w_perturb[i, j] = 0
        
            # save grads
            gradsDict[name] = w_gradsNum
            
            # reset weigth vector
            self.weights[name] = w_0
            
        return gradsDict
    
    def train(
            self, 
            X: np.array, 
            Y: np.array,
            eta: float
        ):
        """
        Parameters
        ----------
        X : seq_len x K one-hot encoded sequence matrix, at t
        Y : seq_len x K one-hot encoded sequence matrix, at t+1
        eta: learning rate
        """
        # get grads from self.computeGrads and update weights
        # w. GD and learning parameter eta
        grads = self.computeGrads(X, Y)
        for key, weight in self.weights.items():
            # clip gradient
            grads[key] = np.clip(grads[key], -5, 5)
            
            # calculate momentum
            self.momentum[key] += np.square(grads[key])
            
            # update weight
            weight -= eta * grads[key] / np.sqrt(self.momentum[key] + 1e-12)

def main():
    # get paths
    home_path = os.path.dirname(os.getcwd())
    data_path = home_path + '\\data\\a4\\'
    plot_path = home_path + '\\a4\\plots\\'
    # results_path = home_path + '\\a4\\results\\'
    
    # get text data
    fname = 'goblet_book.txt'
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
    seq_length = 25
    
    # define X, w. one-hot encoded representations
    data = oneHotEncode(np.array([charToKey[char] for char in data]))
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    
    # init networks
    recurrentNet = recurrentNeuralNetwork(
        K=K,
        m=m,
        sigma=sigma,
        seed=2
    )
    
    # save best weights
    weights_best = recurrentNet.weights.copy()
    
    epoch_n = 0
    print ('\n------EPOCH {}--------\n'.format(epoch_n))
    
    lossHist = []
    loss_smooth = recurrentNet.computeCost(X[0], X[1])
    loss_best = loss_smooth
    
    n = len(X)
    e = 0
    for i in range(2000000):
        recurrentNet.train(X[e], X[e+1], eta=0.1)
        loss = recurrentNet.computeCost(X[e], X[e+1])
        
        loss_smooth = 0.999 * loss_smooth + 0.001 * loss
        if loss_smooth < loss_best:
            weights_best = recurrentNet.weights.copy()
            loss_best = loss_smooth
    
        if (i % 10 == 0) and i > 0:
            lossHist.append(loss_smooth)
            
            if i % 1000 == 0:
                print('Iteration {}, LOSS: {}'.format(i, loss_smooth))
            
        if i % 10000 == 0:
            sequence = recurrentNet.synthesizeText(
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
            recurrentNet.hprev = np.zeros(shape=(m, 1))
            
            epoch_n += 1
            print ('\n------EPOCH {}--------\n'.format(epoch_n))
            
            if epoch_n >= 4:
                break
                
    # plot results
    steps = [step * 10 for step in range(len(lossHist))]
    plt.plot(steps, lossHist, 'r', linewidth=1.5, alpha=1.0, label='Loss')
    plt.xlim(0, steps[-1])
    plt.xlabel('Step')
    plt.ylabel('', rotation=0, labelpad=20)
    plt.title('Smooth loss for $4$ epochs')
    # plt.legend(loc='upper right')
    # plt.savefig(plot_path + 'rnn_loss.png', dpi=200)
    plt.show()
    
    recurrentNet.weights = weights_best
    sequence = recurrentNet.synthesizeText(
        x0=X[e+1][:1], 
        n=1000
    )
    
    # convert to chars and print sequence
    sequence = ''.join([keyToChar[key] for key in sequence])
    print('\nGenerated sequence \n\t {}\n'.format(sequence))
    
if __name__ == '__main__':
    main()


import numpy as np
from misc import softMax

class AdamOpt:
    def __init__(
            self,
            beta1: float,
            beta2: float,
            eps: float,
            weights: list
        ):
        # save init params
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # init dicts for saving moments
        self.m, self.v = {}, {}
        
        # init moments
        for name, weight in weights.items():
            self.m[name] = np.zeros(weight.shape)
            self.v[name] = np.zeros(weight.shape)
            
    def calcMoment(self, beta, moment, grad):
        newMoment = beta * moment + (1 - beta) * grad
        return newMoment
    
    def step(self, weight, grad, t):     
        # update fist moment and correct bias
        self.m[weight] = self.calcMoment(
            self.beta1 ** t,
            self.m[weight], 
            grad
        )
        
        # update second moment and correct bias
        self.v[weight] = self.calcMoment(
            self.beta2 ** t,
            self.v[weight], 
            np.square(grad)
        )
        
        mCorrected = self.m[weight] / (1 - self.beta1 ** t)
        vCorrected = self.v[weight] / (1 - self.beta2 ** t)
        stepUpdate = mCorrected / (np.sqrt(vCorrected) + self.eps)
        return stepUpdate


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
        X : Nxd data matrix

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
        X : Nxd data matrix

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
            Y: np.array, 
            lambd: float
        ) -> float:
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd : regularization parameter
        
        Returns
        -------
        l : cross entropy loss
        J : cost, i.e. cross-entropy loss w. L2-regularization
        """
        # get probs
        P = self.evaluate(X, train=False)
        
        # evaluate loss term
        l = - np.sum(Y.T * np.log(P))
        
        # get regularization term
        r = 0
        
        return l, l + r
    
    def computeAcc(
            self, 
            X: np.array, 
            k: np.array
        ) -> float:
        """
        Parameters
        ----------
        X : Nxd data matrix
        k : Nx1 ground-truth label vector
        Returns
        -------
        acc : accuracy score
        """
        return 0
    
    def computeGrads(
            self, 
            X: np.array, 
            Y: np.array, 
            lambd: float
        ) -> (np.array, np.array):
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd: regularization parameter
        
        Returns
        -------
        W_grads : gradients for weight martix (W)
        b_grads : gradients for bias matrix (b)
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
            lambd: float,
            eps: float,
        ) -> np.array:
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd: regularization parameter
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
                    _, cost1 = self.computeCost(X, Y, lambd)
                
                    # perturb weight vector positively
                    # and compute cost
                    w_tmp = w_0 + w_perturb
                    self.weights[name] = w_tmp
                    _, cost2 = self.computeCost(X, Y, lambd)
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
            lambd: float, 
            eta: float
        ):
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd: regularization parameter
        eta: learning rate
        """
        # get grads from self.computeGrads and update weights
        # w. GD and learning parameter eta
        grads = self.computeGrads(X, Y, lambd)
        for key, weight in self.weights.items():
            # clip gradient
            grads[key] = np.clip(grads[key], -5, 5)
            
            # calculate momentum
            self.momentum[key] += np.square(grads[key])
            
            # update weight
            weight -= eta * grads[key] / np.sqrt(self.momentum[key] + 1e-12)
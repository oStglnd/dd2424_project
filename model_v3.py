
import numpy as np
from misc import softMax, sigmoid

class AdamOpt:
    def __init__(
            self,
            beta1: float,
            beta2: float,
            eps: float,
            layers: list,
        ):
        # save init params
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # init dicts for saving moments
        self.m, self.v = [], []
        
        # init moments
        for idx, layer in enumerate(layers):
            self.m.append({})
            self.v.append({})
            for name, weight in layer.items():
                self.m[idx][name] = np.zeros(weight.shape)
                self.v[idx][name] = np.zeros(weight.shape)
            
    def calcMoment(self, beta, moment, grad):
        newMoment = beta * moment + (1 - beta) * grad
        return newMoment
    
    def step(self, layerIdx, weight, grad, t):     
        # update fist moment and correct bias
        self.m[layerIdx][weight] = self.calcMoment(
            self.beta1 ** t,
            self.m[layerIdx][weight], 
            grad
        )
        
        # update second moment and correct bias
        self.v[layerIdx][weight] = self.calcMoment(
            self.beta2 ** t,
            self.v[layerIdx][weight], 
            np.square(grad)
        )
        
        mCorrected = self.m[layerIdx][weight] / (1 - self.beta1 ** t + self.eps)
        vCorrected = self.v[layerIdx][weight] / (1 - self.beta2 ** t + self.eps)
        stepUpdate = mCorrected / (np.sqrt(vCorrected) + self.eps)
        return stepUpdate
    
class AdaGrad:
    def __init__(
            self,
            eps: float,
            layers: list
        ):
        # save init params
        self.eps = eps
        
        # init dicts for saving moments
        self.m = []
        
        # init moments
        for idx, layer in enumerate(layers):
            self.m.append({})
            for name, weight in layer.items():
                self.m[idx][name] = np.zeros(weight.shape)
    
    def step(self, layerIdx, weight, grad, t):
        
        self.m[layerIdx][weight] += np.square(grad)
        stepUpdate = grad / np.sqrt(self.m[layerIdx][weight] + self.eps)
        
        return stepUpdate
    
class LSTM:
    def __init__(
            self, 
            K_in: int, 
            K_out: int,
            units: list,
            sigma: float,
            optimizer: str,
            embeddings: bool,
            seed: int
        ):
        
        # init seed
        np.random.seed(seed)
        
        # init weight dims
        self.K_in = K_in
        self.K_out = K_out
        self.units = units
        
        # init w2v-mapping
        self.embeddings = embeddings
        if self.embeddings:
            self.keyToVec = None
        
        # init first layer
        m = units[0]
        
        self.layers = []
        self.layers.append({})
        
        
        for key in ['i', 'f', 'e', 'c']:
            self.layers[0]['W_'+key] = np.random.normal(loc=0, scale=2/np.sqrt(m), size=(m, m))
            self.layers[0]['U_'+key] = np.random.normal(loc=0, scale=2/np.sqrt(self.K_in), size=(m, self.K_in))
            self.layers[0]['b_'+key] = np.zeros(shape=(m, 1))
        
        # temp solution
        self.layers[0]['V'] = np.ones(shape=(m, m))
        self.layers[0]['c'] = np.zeros(shape=(m, 1))  
        
        if len(units) > 1:
            # get rest of layers
            for idx, (m1, m2) in enumerate(zip(units[:-1], units[1:])):
                self.layers.append({})
                for key in ['i', 'f', 'e', 'c']:
                    self.layers[idx+1]['W_'+key] = np.random.normal(loc=0, scale=2/np.sqrt(m2), size=(m2, m2))
                    self.layers[idx+1]['U_'+key] = np.random.normal(loc=0, scale=2/np.sqrt(m1), size=(m2, m1))
                    self.layers[idx+1]['b_'+key] = np.zeros(shape=(m2, 1))
                    
                self.layers[idx+1]['V'] = np.ones(shape=(m2, m2))
                self.layers[idx+1]['c'] = np.zeros(shape=(m2, 1))  
            
        # init last V properly
        self.layers[-1]['V'] = np.random.normal(loc=0, scale=2/np.sqrt(self.K_out), size=(self.K_out, units[-1]))
        self.layers[-1]['c'] = np.zeros(shape=(self.K_out, 1))
        
        # init hprev, cprev
        self.hprev = {}
        self.cprev = {}
        for idx, m in enumerate(self.units):
            self.hprev[idx] = np.zeros(shape=(m, 1))
            self.cprev[idx] = np.zeros(shape=(m, 1))


        # init optimizer
        if optimizer.lower() == 'adam':
            self.optimizer = AdamOpt(
                beta1=0.99,
                beta2=0.9999,
                eps=1e-12,
                layers=self.layers
            )
        else:
            self.optimizer = AdaGrad(
                eps=1e-12,
                layers=self.layers
            )


    def synthesizeText(
            self,
            x0: np.array,
            n: int,
            threshold: float,
            temperature: float
        ) -> list:
        
        xList = [x0]
        yList = []
        for _ in range(n):
            p, _, _, _, _, _, _ = self.evaluate(
                xList[-1], 
                train=True, 
                temperature=temperature
            )
            
            # nucleus sampling START
            p_tuples = list(enumerate(p))
            p_tuples.sort(key=lambda x:x[1], reverse=True)
            prob_mass = 0.0
            i = 0
            while prob_mass < (threshold - 10e-4):
                prob_mass += p_tuples[i][1]
                i += 1
 
            id, probabilities = zip(*p_tuples[0:i])  # gets top i number of tokens that make up 95% prob-distr.
            probabilities /= prob_mass        # normalize
            
            idxNext = np.random.choice(
                a=range(self.K_out), 
                p=np.squeeze(p)
            )
            
            if self.embeddings:
                x = self.keyToVec[idxNext][np.newaxis, :]
            else:
                x = np.zeros(shape=(1, self.K_out))
                x[0, idxNext] = 1
                
            xList.append(x)
            yList.append(idxNext)
        
        return yList
        

    def evaluate(
            self, 
            X: np.array,
            train: bool,
            temperature: 1.0
        ) -> np.array:
        """
        Parameters
        ----------
        X : Nxd data matrix

        Returns
        -------
        P : KxN score matrix w. softmax activation
        """
        h, i, f, e, c, cc = [], [], [], [], [], []
        
        for idx, layer in enumerate(self.layers):
            hList = [self.hprev[idx].copy()]
            ccList = [self.cprev[idx].copy()]
            iList, fList, eList, cList = [], [], [], []
            
            # iterate through recurrent block
            for x in X:
                x = x[:, np.newaxis]
                i_t = sigmoid(layer['W_i'] @ hList[-1] + layer['U_i'] @ x + layer['b_i'])
                f_t = sigmoid(layer['W_f'] @ hList[-1] + layer['U_f'] @ x + layer['b_f'])
                e_t = sigmoid(layer['W_e'] @ hList[-1] + layer['U_e'] @ x + layer['b_e'])
                c_t = np.tanh(layer['W_c'] @ hList[-1] + layer['U_c'] @ x + layer['b_c'])
                cc_t = f_t * ccList[-1] + i_t * c_t
                h_t = e_t * np.tanh(cc_t)

                # save vals
                iList.append(i_t)
                fList.append(f_t)
                eList.append(e_t)
                cList.append(c_t)
                ccList.append(cc_t)
                hList.append(h_t)
            
            H = np.hstack(hList)
            X = H.T[1:]
            
            if train:
                self.hprev[idx] = hList[-1]#[1]
                self.cprev[idx] = ccList[-1]#[1]             
        
            i.append(np.hstack(iList))
            f.append(np.hstack(fList))
            e.append(np.hstack(eList))
            c.append(np.hstack(cList))
            cc.append(np.hstack(ccList))
            h.append(np.hstack(hList))
        
        P = softMax(
            self.layers[-1]['V'] @ H[:, 1:] + self.layers[-1]['c'],
            temperature=temperature
        )
        
        # update hprev
        if train:
            return P, i, f, e, c, cc, h
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
            Y: np.array
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
    
    # def computeAcc(
    #         self, 
    #         X: np.array, 
    #         k: np.array
    #     ) -> float:
    #     """
    #     Parameters
    #     ----------
    #     X : Nxd data matrix
    #     k : Nx1 ground-truth label vector
    #     Returns
    #     -------
    #     acc : accuracy score
    #     """
    #     preds = self.predict(X, )
   
    def computeGrads(
            self, 
            X: np.array, 
            Y: np.array
        ) -> (np.array, np.array):
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        
        Returns
        -------
        ....
        """
        # get evaluate and initial g
        P, i, f, e, c, cc, h = self.evaluate(X=X, train=True)
        g = -(Y.T - P)
        
        # get proper list for x
        X = [X] + [h_k.T[1:] for h_k in h[:-1]]
        
        # init grads storage
        grads = []
        for _ in range(len(self.layers)):
            grads.append({})
        
        # get grads for V and c
        grads[-1]['V'] = g @ h[-1].T[1:]
        grads[-1]['c'] = np.sum(g, axis=1)[:, np.newaxis]
        
        # iteratively compute grads for layers w. layer IDX
        for idx, layer in reversed(list(enumerate(self.layers))):
            # compute initial h
            h_grad = g.T[-1] @ layer['V']
            
            # compute grads for initial backward pass
            c_grad = h_grad * e[idx].T[-1] * (1 - np.square(np.tanh(cc[idx].T[-1])))
            
            i_grad = c_grad * c[idx].T[-1] * i[idx].T[-1] * (1 - i[idx].T[-1])
            f_grad = c_grad * cc[idx].T[-2] * f[idx].T[-1] * (1 - f[idx].T[-1])
            ct_grad = c_grad * i[idx].T[-1] * (1 - np.square(c[idx].T[-1]))
            e_grad = h_grad * np.tanh(cc[idx].T[-1]) * e[idx].T[-1] * (1 - e[idx].T[-1])
            
            # init lists
            h_grads = [h_grad]
            c_grads = [c_grad]
            i_grads = [i_grad]
            f_grads = [f_grad]
            e_grads = [e_grad]
            ct_grads = [ct_grad]
            
            # create iterObject
            iterObj = list(zip(
                        g.T[:-1],
                        i[idx].T[:-1],
                        f[idx].T[:-1],
                        f[idx].T[1:],
                        e[idx].T[:-1],
                        c[idx].T[:-1],
                        cc[idx].T[1:-1],
                        cc[idx].T[:-2]
            ))
            
            for gg, ii, ff, ff_prev, ee, cc_old, cc_new, cc_newPrev in reversed(iterObj):
                        
                h_grad = gg @ layer['V']
                h_grad += i_grad @ layer['W_i']
                h_grad += f_grad @ layer['W_f']
                h_grad += e_grad @ layer['W_e']
                h_grad += ct_grad @ layer['W_c']
                
                c_grad = c_grad * ff_prev + h_grad * ee * (1 - np.square(np.tanh(cc_new)))
                
                i_grad = c_grad * cc_old * ii * (1 - ii)
                f_grad = c_grad * cc_newPrev * ff * (1 - ff)
                ct_grad = c_grad * ii * (1 - np.square(cc_old))
                e_grad = h_grad * np.tanh(cc_new) * ee * (1 - ee)
            
                # store grads f. stacking
                h_grads.append(h_grad)
                c_grads.append(c_grad)
                i_grads.append(i_grad)
                f_grads.append(f_grad)
                e_grads.append(e_grad)
                ct_grads.append(ct_grad)
            
            # create grad pairs and get all grads
            grad_pairs = [
                ('i', i_grads),
                ('f', f_grads),
                ('e', e_grads),
                ('c', ct_grads)
            ]
            
            # compute new G
            g = 0
            
            # iterate over key-grad pairs
            for key, grad in grad_pairs:
                grad = np.vstack(grad[::-1]).T
                grads[idx]['W_'+key] = grad @ h[idx].T[:-1]
                grads[idx]['U_'+key] = grad @ X[idx]
                grads[idx]['b_'+key] = np.sum(grad, axis=1)[:, np.newaxis]
        
                g += grad.T @ layer['U_'+key]
            
            g = g.T
        
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
        gradsList = []
        
        for layerIdx, layer in enumerate(self.layers):
            layerDict = {}
            
            for name, weight in layer.items():
                shape = weight.shape
                w_perturb = np.zeros(shape)
                w_gradsNum = np.zeros(shape)
                w_0 = weight.copy()
                
                for i in range(10):
                    for j in range(min(shape[1], 10)):
                # for i in range(shape[0]):
                #     for j in range(shape[1]):
                
                        # add perturbation
                        w_perturb[i, j] = eps
                        
                        # perturb weight vector negatively
                        # and compute cost
                        w_tmp = w_0 - w_perturb
                        self.layers[layerIdx][name] = w_tmp
                        _, cost1 = self.computeCost(X, Y)
                    
                        # perturb weight vector positively
                        # and compute cost
                        w_tmp = w_0 + w_perturb
                        self.layers[layerIdx][name] = w_tmp
                        _, cost2 = self.computeCost(X, Y)
                        lossDiff = (cost2 - cost1) / (2 * eps)
                        
                        # get numerical grad f. W[i, j]
                        w_gradsNum[i, j] = lossDiff
                        w_perturb[i, j] = 0
            
                # reset weigth vector
                self.layers[layerIdx][name] = w_0
                layerDict[name] = w_gradsNum
            gradsList.append(layerDict)
            
        return gradsList
    
    def train(
            self, 
            X: np.array, 
            Y: np.array,
            t: int,
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
        gradsList = self.computeGrads(X, Y)
        
        for idx, grads in enumerate(gradsList):
            for key, grad in grads.items():
                grad = np.clip(grad, -5, 5)
                
                # calculate momentum and step update
                stepUpdate = self.optimizer.step(
                    idx, 
                    key, 
                    grad, 
                    t
                )
        
                # update weight
                self.layers[idx][key] -= eta * stepUpdate
        
        # for idx, layer in enumerate(self.layers): 
        #     for key, weight in layer.items():
                
        #         print(idx, key)
                
        #         print(idx, key)
        #         # clip gradient
        #         grad = grads[idx][key]
        #         grad = np.clip(grad, -5, 5)
                
        #         # calculate momentum
        #         self.momentum[idx][key] += np.square(grad)
                
        #         # update weight
        #         weight -= eta * grad / np.sqrt(self.momentum[idx][key] + 1e-12)
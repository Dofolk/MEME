#%%
import numpy as np
import collections
import random
import math
from math import sqrt

#%%
#set up the global variables
if 1:
    A = {'A':0,'T':1,'C':2,'G':3}
else:
    A = {}
L = len(A)
#%%
Y = ['TGCTGGATAAGAATGTTTTAGCAATCTCTTT', 'TCAGCGAAAAAAATTAAAGCGCAAGATTGTT', 'GCGACAACCGGAATATGAAAGCAAAGCGCAG']
W = 6

# %%

class MEME:
    def __init__(self, Y, W):
        # the basic variables, parameters in the MEME
        # the symbols are corresponding to the article
        # the small z stores the information of big Z from motif for each subsequences and 
        # the storage position is depending on the start position index of the subsequence

        self.Y = Y
        self.N = len(Y)
        self.l = [len(seqs) for seqs in Y]
        self.W = W
        self.X, self.total_freq = self.data_div(Y, W)
        self.n = len(self.X)
        self.Z = None
        self.z = np.zeros([self.N, max(self.l)],dtype = 'float')
        #self.z = [ [ [self.Z[idx][0] for idx in range(self.n)] for i in range(self.l[j])] for j in range(self.N)]
        
        self.f = np.zeros((self.W, L))
        self.f_0 = np.zeros((L, ))
        #self.f = abs(np.random.rand())
        
        self.theta1 = self.f
        self.theta2 = self.f_0

        self.lamb1  = None
        self.lamb2 = None
        self.I = None
        self.I_str = None
        self.var_init()
        self.theta = [self.theta1, self.theta2]
        self.lamb = [self.lamb1, self.lamb2]

    
    def var_init(self):
        # initialize the variables, parameters
        # it will initialize: lambda 1, lambda 2, indicator function I(k,a)

        self.Z = np.repeat(np.array([[1,0]]), self.n, axis = 0)

        lamb_range_m = min(sqrt(self.N)/self.n, 1/(2*self.W))
        lamb_range_M = max(sqrt(self.N)/self.n, 1/(2*self.W))
        self.lamb1 = random.uniform(lamb_range_m, lamb_range_M)
        self.lamb2 = 1 - self.lamb1
        
        self.I, self.I_str = self.indicator()
        
        idx = 0
        for seq in range(self.N):
            length = self.l[seq]
            val = np.pad(np.array([self.Z[idx][0] for idx in range(idx, idx + length - self.W + 1, 1)])\
                         , (0, self.W - 1), mode = 'constant').reshape(length,)
            self.z[seq][0:length] = val

        return None
    
    def data_div(self, Y, W):
        # divide the input data into subsequences
        # correspond to the X and Y in the article
        # also calculate the freq of each nucletide appearence

        X = list()
        C = dict( zip(A.keys(), [0]*L) )
        for i in range(len(Y)):
            X += [ Y[i][j:j+W] for j in range( len(Y[i]) - W + 1) ]
            for key in C:
                C[key] += Y[i].count(key)

        return X, C

    def indicator(self):
        # contribute the indicator function matrix results
        
        I_total = np.zeros((self.n, self.W, L), dtype = 'int')
        I_str_idx = np.zeros((self.n, self.W), dtype = 'int')
        
        for subseq in range(self.n):
            for pos in range(self.W):
                I_total[subseq][pos][ A[self.X[subseq][pos] ] ] = 1
                I_str_idx[subseq][pos] = A[self.X[subseq][pos] ]

        return I_total, I_str_idx 

    def condi_distributions(self, theta):
        # calculate the conditional distribution p(Xi | theta_j)
        # eq(7),(8) in the article

        p_Xi_1 = np.ones((self.n, 1), dtype = 'float')
        p_Xi_2 = np.ones((self.n, 1), dtype = 'float')
        f = theta[0]
        f_0 = theta[1]

        for subseq in range(self.n):
            for pos in range(self.W):
                p_Xi_1[subseq] *= f[pos][self.I_str[subseq][pos]]
                p_Xi_2[subseq] *= f_0[self.I_str[subseq][pos]]
        
        return p_Xi_1, p_Xi_2

    def E_step(self, theta, lamb):
        # the expectation step will update Zij

        lamb1 = lamb[0]
        lamb2 = lamb[1]
        p_Xi_1, p_Xi_2 = self.condi_distributions(theta)
        p1 = p_Xi_1 * lamb1
        p2 = p_Xi_2 * lamb2
        summation = p1 + p2

        Z_ij = np.divide( np.concatenate((p1,p2), axis = 1), summation)
        E_val = np.sum( np.multiply( Z_ij, np.concatenate((np.log(p1),np.log(p2)), axis = 1 ) ) )
        
        self.update_var(Z_ij, theta, lamb)

        return E_val
    
    def M_step(self, Z):
        return
    
    def update_var(self, Z, theta, lamb):
        # update variables, parameters after each step or operation

        self.Z = Z
        self.theta = theta
        self.lamb = lamb
        
        idx = 0
        for seq in range(self.N):
            length = self.l[seq]
            val = np.pad(np.array([self.Z[idx][0] for idx in range(idx, idx + length - self.W + 1, 1)])\
                         , (0, self.W - 1), mode = 'constant').reshape(length,)
            self.z[seq][0:length] = val

        return
        

    def iter(self):
        return
# %%

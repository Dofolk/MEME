#%%
import numpy as np
import collections
import random
import math
from math import sqrt
from sklearn import preprocessing

#%%
#set up the global variables
if 1:
    A = {'A':0,'T':1,'C':2,'G':3}
else:
    A = {}
L = len(A)

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
        self.X, self.total_freq, self.subseqs_amount_list = self.data_div(Y, W)
        self.n = len(self.X)
        self.Z = None
        self.z = None
        
        self.f = None
        self.f_0 = None

        self.lamb1  = None
        self.lamb2 = None
        self.I = None
        self.I_str = None
        self.I_subseq_freq = None
        self.var_init()
        self.theta = [self.f, self.f_0]
        self.lamb = [self.lamb1, self.lamb2]

        self.erase = np.ones(self.n)

    
    def var_init(self):
        # initialize the variables, parameters
        # it will initialize: big Z, small z, lambda 1, lambda 2, f, f_0

        self.Z = np.repeat(np.array([[1,0]]), self.n, axis = 0)
        if self.n != sum(self.subseqs_amount_list):
            print('Error')
        else:
            self.z = list()
            z_idx = 0
            for i in range(self.n):
                for j in self.subseqs_amount_list:
                    self.z[i][j] = self.Z[z_idx][0]
                    z_idx += 1
            for i in range(len(self.z)):
                self.z[i] = preprocessing.normalize(self.z[i])
                

        lamb_range_m = min(sqrt(self.N)/self.n, 1/(2*self.W))
        lamb_range_M = max(sqrt(self.N)/self.n, 1/(2*self.W))
        self.lamb1 = random.uniform(lamb_range_m, lamb_range_M)
        self.lamb2 = 1 - self.lamb1
        
        self.I, self.I_str, self.I_subseq_freq = self.indicator()
        
        idx = 0
        for seq in range(self.N):
            length = self.l[seq]
            val = np.pad(np.array([self.Z[idx][0] for idx in range(idx, idx + length - self.W + 1, 1)])\
                         , (0, self.W - 1), mode = 'constant').reshape(length,)
            self.z[seq][0:length] = val
        
        f = np.array([random.uniform(0,1) for _ in range(L * self.W)]).reshape(self.W, L)
        for pos in range(self.W):
            f[pos] = preprocessing.normalize(f[pos])
        self.f = f
        self.f_0 = preprocessing.normalize(np.array([random.unifrom(0,1) for _ in range(L)]).reshape(1, L))

        return None
    
    def data_div(self, Y, W):
        # divide the input data into subsequences
        # correspond to the X and Y in the article
        # also calculate the freq of each nucletide appearence

        X = list()
        C = dict( zip(A.keys(), [0]*L) )
        amount = list()
        for i in range(len(Y)):
            subsequences = [ Y[i][j:j+W] for j in range( len(Y[i]) - W + 1) ]
            X += subsequences
            amount.append(len(subsequences))
            for key in C:
                C[key] += Y[i].count(key)

        return X, C, amount

    def indicator(self):
        # contribute the indicator function matrix results
        # I_str_idx 
          # type : np.array
          # size : n x W
          # meaning : transfer each subsequences' letters into integers
        # I_str_count
          # type : list of dictionary
          # size : n dict() elements for each elements contain counter info.
          # meaning : counts each subsequences letters appearance
        
        I_total = np.zeros((self.n, self.W, L), dtype = 'int')
        I_str_idx = np.zeros((self.n, self.W), dtype = 'int')
        I_str_count = list()
        
        for subseq in range(self.n):
            for pos in range(self.W):
                I_total[subseq][pos][ A[self.X[subseq][pos] ] ] = 1
                I_str_idx[subseq][pos] = A[self.X[subseq][pos] ]
            I_str_count.append(collections.Counter(list(I_str_idx)))

        return I_total, I_str_idx, I_str_count

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

        self.E_step_update(Z_ij)

        return E_val
    
    def E_step_update(self, Z):
        self.Z = Z
        return

    def M_step(self, Z):
        lamb1 = 0
        lamb2 = 0
        
        # update lambda value
        for i in range(self.n):
            lamb1 += Z[i][0]
            lamb2 += Z[i][1]
        lamb1 /= self.n
        lamb2 /= self.n

        # update theta value
        c_0k = np.zeros([1,L])
        c_jk = np.zeros([self.W, L])
        
        
        for k in range(L):
            for i in range(self.n):
                freq = self.I_subseq_freq[i]
                c_0k[0][k] += Z[i][1]*freq[k]
            for j in range(self.W):
                val = 0
                for i in range(self.n):
                    if self.I_str[i][j] is k:
                        val += self.erase[i]*self.Z[i][0]
                c_jk[j][k] = val
        
        self.M_step_update([lamb1,lamb2], [c_0k,c_jk])

        return
    
    def M_step_update(self, lamb, c):

        # update lambda value
        self.lamb1, self.lamb2, self.lamb = lamb[0], lamb[1], lamb

        # update theta value
        self.f_0 = c[0]/sum(c[0])
        for pos in range(self.W):
            self.f[pos] = c[pos]/sum(c[pos])
        self.theta = [self.f, self.f_0]

        return    

    def iter(self):
        return
# %%

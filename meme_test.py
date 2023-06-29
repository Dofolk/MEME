#%%
import numpy as np
import collections
A = {'A':0,'T':1,'C':2,'G':3}

#%%
def data_div(Y,W):
    X = list()
    for i in range(len(Y)):
        X += [ Y[i][j:j+W] for j in range( len(Y[i]) - W + 1) ]
    return X

def indicator_I(X):
        W = len(X)
        I_pos = np.zeros((W, L))
        I_str = np.zeros((W, ),dtype='int')
        for i in range(W):
            I_pos[i][A[X[i]]] = 1
            I_str[i] = A[X[i]]

        return I_pos, I_str

#%%
Y = ['ATCGCGG']
N = len(Y)
l = [len(seqs) for seqs in Y]
W = 5

L = len(A)
X = data_div(Y, W)
n = len(X)
Z = [np.array([1,0]) for _ in range(n)]
z = [ [ [Z[idx][0] for idx in range(n)] for i in range(l[j])] for j in range(N)]
f = np.zeros((W, L))
f_0 = np.zeros((L, ))
theta1 = f
theta2 = f_0
lamb1 = 1
lamb2 = 1 - lamb1
theta = [theta1, theta2]
lamb = [lamb1, lamb2]

#%%
p_Xi_1 = np.ones((n, 1))
p_Xi_2 = np.ones((n, 1))
f_0 = theta[1]
f = theta[0]

for i in range(n):
    _, I_str = indicator_I(X[i])
    print(i)
    for j in range(W):
        p_Xi_1[i] *= f[j][I_str[j]]
    alphabet_count = collections.Counter(I_str)
    for k in range(L):
        p_Xi_2[i] *= (f_0[k] ** alphabet_count[k])
# %%

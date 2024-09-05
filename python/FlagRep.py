import numpy as np
import scipy
from matplotlib import pyplot as plt

def chordal_distance(X, Y, Bs_x, Bs_y):

    k = len(Bs_x)

    dist = 0

    for i in range(k):
        id_x = Bs_x[i]
        id_y = Bs_y[i]
        Xi = X[:,id_x]
        Yi = Y[:,id_y]
        mm = np.min([len(Bs_x[i]), len(Bs_y[i])])
        dist = dist + np.sqrt(mm - np.trace(Xi.T @ Yi @ Yi.T @ Xi))

    return dist



def FlagRep(D: np.array, Aset: list, eps_rank: float = 1e-8) -> tuple:

    n,_ = D.shape


    # output flag
    X = []

    # get the number of As
    k = len(Aset)

    # for feature indices
    Bset = []

    # first part of the flag
    Bset.append(Aset[0])
    B = D[:,Bset[0]]
    C = B
    U,S,_ = np.linalg.svd(C, full_matrices=False)
    print(S)
    X.append(U[:,S>eps_rank])
    P = np.eye(n) - X[-1] @ X[-1].T

    m = np.zeros((k,1))

    # m[0] = np.linalg.matrix_rank(C)
    m[0] = X[-1].shape[1]

    # the rest of the flag
    for i in range(1,k):
        # print(i)
        Bset.append(np.setdiff1d(Aset[i],Aset[i-1]))
        B = D[:,Bset[i]]
        C = P @ B
        U,S,_ = np.linalg.svd(C, full_matrices=False)
        X.append(U[:,S>eps_rank])
        P = (np.eye(n) - X[-1] @ X[-1].T) @ P
        # m[i] = np.linalg.matrix_rank(C)
        m[i] = X[-1].shape[1]

    # translate to stiefel manifold representative n x n_k
    X = np.hstack(X)

    # compute the flag type (n_1,n_2,...,n_k)
    flag_type = np.cumsum(m).astype(int)

    return X, flag_type
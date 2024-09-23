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
        sin_sq = mm - np.trace(Xi.T @ Yi @ Yi.T @ Xi)
        if np.isclose(sin_sq,0):
            sin_sq = 0
        elif sin_sq < 0:
            print('sine squared less than 0')
            print(sin_sq)
        
        dist = dist + np.sqrt(sin_sq)

    return dist


def truncate_svd(C: np.array, eps_rank: float = 1e-8, zero_tol: float = 1e-8) -> np.array:
    U,S,_ = np.linalg.svd(C, full_matrices=False)

    # try 1
    nnz_ids = ~np.isclose(S, 0, atol=zero_tol)
    S = S[nnz_ids]
    U = U[:,nnz_ids]
    s_prop = np.cumsum(S**2)/np.sum(S**2)
    good_idx = s_prop<=eps_rank
    U = U[:,good_idx]

    # try 2 a la https://arxiv.org/pdf/1305.5870
    # m, n = C.shape
    # beta = m/n
    # lambda_ast = np.sqrt(2*(beta+1) + 8*beta/(beta + 1 + np.sqrt(beta**2 + 14*beta + 1)))
    # cutoff = lambda_ast*np.sqrt(n)
    # good_idx = S >= cutoff
    # U[:,good_idx]

    # try 3
    # nnz_ids = ~np.isclose(S, 0, atol=zero_tol)
    # S = S[nnz_ids]
    # U = U[:,nnz_ids]

    return U


def FlagRep(D: np.array, Aset: list, eps_rank: float = 1e-8, zero_tol: float = 1e-8) -> tuple:

    '''
    Maybe try to remove truncation.
    '''

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
    U = truncate_svd(C, eps_rank, zero_tol)
    X.append(U)
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
        C[np.isclose(C, 0, atol=zero_tol)] = 0
        if np.all(C == 0):
            m[i] = 0
        else:
            U = truncate_svd(C, eps_rank, zero_tol)
            X.append(U)
            P = (np.eye(n) - X[-1] @ X[-1].T) @ P
            # m[i] = np.linalg.matrix_rank(C)
            m[i] = X[-1].shape[1]

    # translate to stiefel manifold representative n x n_k
    X = np.hstack(X)
    if X.shape[1] > n:
        print(f'error {np.cumsum(m).astype(int)}')
        X = X[:,:n]

    # compute the flag type (n_1,n_2,...,n_k)
    m = m[m != 0] # remove 0s
    flag_type = np.cumsum(m).astype(int)

    return X, flag_type
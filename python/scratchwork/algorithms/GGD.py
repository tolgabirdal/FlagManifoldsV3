import numpy as np
from numpy.linalg import svd, norm

def ggd(X, s, maxiter, d, opt, true_sub=None):
    """
    GGD: Geodesic Gradient Descent
    
    X:        N x D data matrix
    s:        initial step size
    maxiter:  number of iterations
    d:        subspace dimension
    opt:      0 for s/sqrt(k) step size, 
              1 for line search step size,
              2 for shrinking s
    true_sub: true subspace (for convergence plot)

    Implementation of geodesic gradient descent from the paper 
    'A Well Tempered Landscape for Non-convex Robust Subspace Recovery'
    https://arxiv.org/abs/1706.03896
    """
    
    if true_sub is None:
        true_sub = np.linalg.qr(np.random.randn(X.shape[1], d))[0]

    N, D = X.shape
    U0, S0, V0 = svd(X, full_matrices=False)
    Vk = V0.T[:, :d]
    
    tol = 1e-5
    Vprev = Vk
    seq_dist = 1
    k = 1
    conv = []

    while k < maxiter and seq_dist > tol:
        # calculate gradient
        X_dot_vk = X @ Vk
        dists = np.sqrt(np.sum((X.T - Vk @ (Vk.T @ X.T)) ** 2, axis=0))
        dists[dists < 1e-12] = np.inf  # points on subspace contribute 0
        scale = X_dot_vk / dists[:, np.newaxis]

        derFvk = X.T @ scale
        gradFvk = derFvk - Vk @ (Vk.T @ derFvk)

        # SVD for geodesic calculation
        U, Sigma, W = svd(gradFvk, full_matrices=False)
        W = W.T

        if opt == 1:
            # line search
            step = s
            cond = True
            curr_cost = cost(Vk, X)
            while cond:
                Vkt = Vk @ W @ np.diag(np.cos(Sigma * step)) + U @ np.diag(np.sin(Sigma * step))
                if cost(Vkt, X) < curr_cost or step <= 1e-16:
                    Vk = Vkt
                    cond = False
                else:
                    step /= 2
        elif opt == 2:
            # shrinking s
            step = s
            Vkt = Vk @ W @ np.diag(np.cos(Sigma * step)) + U @ np.diag(np.sin(Sigma * step))
            Vk = Vkt
            if k % 50 == 0:
                s /= 10
        else:
            # 1/sqrt(k)
            step = s / np.sqrt(k)
            Vkt = Vk @ W @ np.diag(np.cos(Sigma * step)) + U @ np.diag(np.sin(Sigma * step))
            Vk = Vkt

        # calculate maximum principal angle between A and true subspace (for plots)
        A = true_sub.T @ Vk
        sv = svd(A, full_matrices=False, compute_uv=False)
        sv[sv >= 1] = 0
        sv = np.arccos(sv)
        conv.append(np.max(sv))

        k += 1
        seq_dist = calc_sdist(Vk, Vprev)
        Vprev = Vk
    
    #nate added this
    Vk = np.linalg.qr(Vk)[0][:,:d]

    return Vk


def cost(V, X):
    """Calculate cost function from the paper"""
    return np.sum(np.sqrt(np.sum((X.T - V @ (V.T @ X.T)) ** 2, axis=0)))


def calc_sdist(Vk, Vprev):
    """Calculate sequential distance between Vk and Vprev"""
    return norm(Vk - Vprev)


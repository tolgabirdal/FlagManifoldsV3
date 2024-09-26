import numpy as np

from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

from collections import Counter

'''
Useful functions... almost all code from https://arxiv.org/pdf/2307.15348
'''



def ll(model, eigval, n):
    """ Compute the maximum log-likelihood of a PSA model from the sample eigenvalues.
    eigval must be sorted in decreasing order.
    """
    p = np.sum(model)
    q_list = (0,) + tuple(np.cumsum(model))
    eigval_mle = np.concatenate([[np.mean(eigval[qk:qk_])] * gamma_k for (qk, qk_, gamma_k) in zip(q_list[:-1], q_list[1:], model)])
    return - (n / 2) * (p * np.log(2 * np.pi) + np.sum(np.log(eigval_mle)) + p)

def kappa(model):
    """ Compute the number of free parameters of a PSA model.
    """
    p = np.sum(model)
    kappa_mu = p
    kappa_eigvals = len(model)
    kappa_eigenspaces = int(p * (p - 1) / 2 - np.sum(np.array(model) * (np.array(model) - 1) / 2))
    return kappa_mu + kappa_eigvals + kappa_eigenspaces

def bic(model, eigval, n):
    """ Compute the Bayesian Information Criterion (BIC) of a PSA model from the sample eigenvalues.
    eigval must be sorted in decreasing order.
    """
    return kappa(model) * np.log(n) - 2 * ll(model, eigval, n)

def aic(model, eigval, n):
    """ Compute the Akaike Information Criterion (AIC) of a PSA model from the sample eigenvalues.
    eigval must be sorted in decreasing order.
    """
    return 2 * kappa(model) - 2 * ll(model, eigval, n)

def evd(X, plot_scree=False):
    """ Perform the eigenvalue decomposition (EVD) of the sample covariance matrix of X.
    Return sample eigenvalues and eigenvectors with decreasing amplitude.
    """
    n, p = X.shape
    mu = np.mean(X, axis=0)
    S = 1 / n * ((X - mu).T @ (X - mu))
    eigval, eigvec = np.linalg.eigh(S)
    eigval, eigvec = np.flip(eigval, -1), np.flip(eigvec, -1)
    if plot_scree:
        fig = plt.figure()
        plt.bar(np.arange(1, p + 1), eigval, color='k')
        plt.title("Eigenvalue scree plot")
        plt.show()
    return eigval, eigvec

def candidate_models_hierarchical(eigval, distance="relative", linkage="single"):
    """ Perform a hierarchical clustering of the sample eigenvalues for model selection.
    """
    p = len(eigval)
    updiag = np.array([[j == (i + 1) for j in range(p)] for i in range(p)]).astype('int')
    if distance == "absolute":
        metric = "euclidean"
        Z = eigval[:, None]
    elif distance == "relative":
        metric = "precomputed"
        Z = np.zeros((p, p))
        for j in range(p-1):
            Z[j, j+1] = (eigval[j] - eigval[j+1]) / eigval[j]
        Z = Z + Z.T
    else:
        raise NotImplementedError()
    clustering = AgglomerativeClustering(connectivity=updiag+updiag.T, metric=metric, linkage=linkage).fit(Z)
    merges = clustering.children_
    models = [[1,] * p]
    nodes_locations = [i for i in range(p)]
    for j, merge in enumerate(merges):
        model = models[-1].copy()
        k = min(nodes_locations.index(merge[0]), nodes_locations.index(merge[1]))
        model[k] += model[k+1]
        del model[k+1]
        models.append(model)
        nodes_locations[k] = p + j
        del nodes_locations[k+1]
    return models

def model_selection(X, candidate_models, criterion="bic"):
    """ Perform model selection by minimizing a criterion (AIC or BIC) among a family of candidate models.
    eigval must be sorted in decreasing order.
    """
    n, p = X.shape
    eigval, _ = evd(X)
    model_best = None; crit_best = np.inf
    for model in candidate_models:
        if criterion == "bic":
            crit_model = bic(model, eigval, n)
        elif criterion == "aic":
            crit_model = aic(model, eigval, n)
        else:
            raise NotImplementedError
        if crit_model < crit_best:
            model_best = model
            crit_best = crit_model
    return model_best, crit_best

def model_selection_eval(X, eigval, candidate_models, criterion="bic"):
    """ Perform model selection by minimizing a criterion (AIC or BIC) among a family of candidate models.
    eigval must be sorted in decreasing order.
    """
    n, p = X.shape
    model_best = None; crit_best = np.inf
    for model in candidate_models:
        if criterion == "bic":
            crit_model = bic(model, eigval, n)
        elif criterion == "aic":
            crit_model = aic(model, eigval, n)
        else:
            raise NotImplementedError
        if crit_model < crit_best:
            model_best = model
            crit_best = crit_model
    return model_best, crit_best


def most_common_fl_type(fl_types):
    # Most common fl_type
    tuples = [tuple(sublist) for sublist in fl_types]
    # Use Counter to count occurrences of each tuple
    count = Counter(tuples)

    # Find the most common tuple
    most_common_tuple = count.most_common(1)[0][0]

    # Convert it back to a list
    mean_f_type = list(most_common_tuple)

    return mean_f_type
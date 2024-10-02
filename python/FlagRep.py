import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator

class FlagRep(BaseEstimator):
    def __init__(self, Aset: list = [], flag_type: list = [], eps_rank: float = 1, zero_tol: float = 1e-8):

        self.Aset_ = Aset
        self.eps_rank_ = eps_rank
        self.zero_tol_ = zero_tol
        self.flag_type_ = flag_type
        self.D_ = np.array([])

        # Check if flag_type matches the size of Aset
        if len(self.flag_type_) != len(self.Aset_) and len(self.flag_type_) > 0:
            raise ValueError('flag_type and Aset lengths are not equal')

    def flag_type(self):
        return self.flag_type_

    def fit_transform(self, D):
        """
        Apply the transformation to the data.

        Parameters:
        D: array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns:
        X: array-like, shape (n_samples, n_features)
            Transformed version of the input data.
        """
        self.D_ = D

        self.n_,self.p_ = self.D_.shape

        # Ensure the matrix is not too large to handle
        if self.n_ > 10000 or self.p_ > 10000:
            raise MemoryError("Input matrix is too large. Consider reducing the size.")
        
        if self.n_ < self.p_:
            raise ValueError("D must be a tall-skinny matrix (n > p). You passed n <= p.")


        # output flag
        X = []

        # get the number of As
        k = len(self.Aset_)

        # for feature indices
        Bset = []

        
        # first part of the flag
        Bset.append(self.Aset_[0])
        B = self.D_[:,Bset[0]]
        C = B

        if len(self.flag_type_) > 0:
            U = self.truncate_svd(C, n_vecs = self.flag_type_[0])
        else:
            U = self.truncate_svd(C)

        X.append(U)

        P = np.eye(self.n_) - X[-1] @ X[-1].T
        m = np.zeros((k,1))
        m[0] = X[-1].shape[1]

        # the rest of the flag
        for i in range(1,k):
            Bset.append(np.setdiff1d(self.Aset_[i],self.Aset_[i-1]))
            B = self.D_[:,Bset[i]]
            C = P @ B
            C[np.isclose(C, 0, atol=self.zero_tol_)] = 0
            if np.all(C == 0):
                m[i] = 0
            else:
                if len(self.flag_type_) > 0:
                    U = self.truncate_svd(C, n_vecs = self.flag_type_[0])
                else:
                    U = self.truncate_svd(C)

                X.append(U)

                P = (np.eye(self.n_) - X[-1] @ X[-1].T) @ P
                m[i] = X[-1].shape[1]

        # translate to stiefel manifold representative n x n_k
        X = np.hstack(X)
        if X.shape[1] > self.n_:
            print(f'error {np.cumsum(m).astype(int)}')
            X = X[:,:self.n_]

        # compute the flag type (n_1,n_2,...,n_k)
        m = m[m != 0] # remove 0s
        self.flag_type_ = np.cumsum(m).astype(int)

        return X

    def inverse_transform(self, X):
        """
        Apply the inverse transformation to the data.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            The transformed data to inverse transform.

        Returns:
        X_original: array-like, shape (n_samples, n_features)
            Data in its original form before transformation.
        """
        X_original = np.zeros((self.n_, self.p_))

        P = np.eye(self.n_)

        # loop through flag
        for i in range(len(self.Aset_)):
            
            if i < len(self.flag_type_):
                n_i = self.flag_type_[i]
                flag_i = i
            else:
                print('number of subspaces in flag shorter than number feature sets')
                print('... estimating reconstruction using final part of flag')

            if flag_i == 0:
                n_im1 = 0
                Bset_i = self.Aset_[i]
                P = X[:,n_im1: n_i] @ X[:,n_im1: n_i].T
                X_original[:, Bset_i] = P @ self.D_[:,Bset_i]

            else:
                n_im1 = self.flag_type_[i-1]
                Bset_i = np.setdiff1d(self.Aset_[i],self.Aset_[i-1])
                P = (np.eye(self.n_) - X[:,n_im1: n_i] @ X[:,n_im1: n_i].T) @ P
                X_original[:, Bset_i] = (P + X[:,n_im1: n_i] @ X[:,n_im1: n_i].T) @ self.D_[:,Bset_i]

        return X_original

    def truncate_svd(self, C: np.array, n_vecs: int = 0) -> np.array:
        U,S,_ = np.linalg.svd(C, full_matrices=False)
        if n_vecs > 0:
            U = U[:,:n_vecs]
        else:
            nnz_ids = ~np.isclose(S, 0, atol=self.zero_tol_)
            U = U[:,nnz_ids]
            S = S[nnz_ids]

            s_prop = np.cumsum(S**2)/np.sum(S**2)
            good_idx = s_prop<=self.eps_rank_
            U = U[:,good_idx]

        return U
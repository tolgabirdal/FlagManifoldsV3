import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator

class FlagRep(BaseEstimator):
    def __init__(self, Aset: list = [], flag_type: list = [], 
                 eps_rank: float = 1, zero_tol: float = 1e-10,
                 solver = 'svd'):

        self.Aset_ = Aset
        self.eps_rank_ = eps_rank
        self.zero_tol_ = zero_tol
        self.flag_type_ = flag_type
        self.D_ = np.array([])
        self.solver_ = solver

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
        if self.n_ > 10000:# or self.p_ > 10000:
            raise MemoryError("Input matrix is too large. Consider reducing the size.")


        # if self.n_ < self.p_:
        #     raise ValueError("D must be a tall-skinny matrix (n > p). You passed n <= p.")


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
            U = self.get_basis(C, n_vecs = self.flag_type_[0])
        else:
            U = self.get_basis(C)

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
                    U = self.get_basis(C, n_vecs = self.flag_type_[i]-self.flag_type_[i-1])
                else:
                    U = self.get_basis(C)

                X.append(U)

                if i < k-1:
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
        if self.solver_ != 'svd':
            raise ValueError('Inverse transform only supported for solver = svd')

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
                Pxi = X[:,n_im1: n_i] @ X[:,n_im1: n_i].T
                P = Pxi + (np.eye(self.n_) - Pxi) @ P
                X_original[:, Bset_i] = P @ self.D_[:,Bset_i]

        return X_original
   
    def decompose(self, D: np.array = np.empty([])):

        if self.solver_ != 'svd':
            raise ValueError('Decomposition only supported for solver = svd')

        X = self.fit_transform(D)

        n_k = X.shape[1]

        R = np.zeros((n_k,self.p_))

        for i in range(len(self.flag_type_)):

            if i == 0:
                n_im1 = 0
            else:
                n_im1 = self.flag_type_[i-1]
            n_i = self.flag_type_[i]

            for j in range(i,len(self.Aset_)):

                if j == 0:
                    Bset_j = self.Aset_[j]
                    p_jm1 = 0
                else:
                    Bset_j = np.setdiff1d(self.Aset_[j],self.Aset_[j-1])
                    p_jm1 = len(self.Aset_[j-1])
                p_j = len(self.Aset_[j])

                Bij = X[:,n_im1:n_i].T @ D[:, Bset_j]
                R[n_im1:n_i,p_jm1:p_j] = Bij
        
        return X, R

    def objective_value(self, X: np.array, D: np.array = np.empty([]), fl_type: list = []) -> float:
        if len(fl_type) == 0:
            fl_type = self.flag_type_
        
        if len(D) == 0:
            D = self.D_

        obj_val = 0
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
                P = np.eye(self.n_) - X[:,n_im1: n_i] @ X[:,n_im1: n_i].T

            else:
                n_im1 = self.flag_type_[i-1]
                Bset_i = np.setdiff1d(self.Aset_[i],self.Aset_[i-1])
                P = (np.eye(self.n_) - X[:,n_im1: n_i] @ X[:,n_im1: n_i].T) @ P
            
            obj_val += np.linalg.norm(P @ D[:,Bset_i])**2
        
        return obj_val

    def truncate_svd(self, C: np.array, n_vecs: int = 0) -> np.array:
        
        U,S,_ = np.linalg.svd(C, full_matrices=False)
        if n_vecs > 0:
            U = U[:,:n_vecs]
        else:
            nnz_ids = ~np.isclose(S, 0, atol=self.zero_tol_)
            U = U[:,nnz_ids]
            S = S[nnz_ids]

            s_prop = np.cumsum(S**2)/np.sum(S**2)
            good_idx = s_prop<=self.eps_rank_ + self.zero_tol_
            U = U[:,good_idx]

        return U

    def truncate_qr(self, C, n_vecs: int = 0) -> np.array:
        Q,R,_ = scipy.linalg.qr(C, pivoting = True)

        if n_vecs > 0:
            Q = Q[:,:n_vecs]

        else:
            nonzero_rows = ~np.all(np.isclose(R, 0, atol=self.zero_tol_), axis=1)
            nonzero_row_indices = np.where(nonzero_rows)[0]
            Q = Q[:,nonzero_row_indices]

        return Q

    def get_basis(self, C, n_vecs: int = 0):
        if self.solver_ == 'svd':
            if len(self.flag_type_) > 0:
                U = self.truncate_svd(C, n_vecs = n_vecs)
            else:
                U = self.truncate_svd(C)

        elif self.solver_ == 'qr':
            if len(self.flag_type_) > 0:
                U = self.truncate_qr(C, n_vecs = n_vecs)
            else:
                U = self.truncate_qr(C)
        
        else:
            raise ValueError('Solver must be either qr or svd')
        
        return U


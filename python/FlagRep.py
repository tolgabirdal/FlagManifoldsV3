import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from GGD import ggd

class FlagRep(BaseEstimator):
    def __init__(self, Aset: list = [], flag_type: list = [], 
                 eps_rank: float = 1, zero_tol: float = 1e-10,
                 solver = 'svd', plot_eigs = False):

        self.Bset_ = [Aset[0]]+[np.setdiff1d(Aset[i],Aset[i-1])for i in range(1,len(Aset))]
        # self.Bset_ = [Aset[-1]]+[np.setdiff1d(Aset[-1],Aset[i])for i in range(len(Aset)-1)]
        self.eps_rank_ = eps_rank
        self.zero_tol_ = zero_tol
        self.flag_type_ = flag_type
        self.D_ = np.array([])
        self.solver_ = solver
        self.plot_eigs_ = plot_eigs

        # Check if flag_type matches the size of Aset
        if len(self.flag_type_) != len(self.Bset_) and len(self.flag_type_) > 0:
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

        # get the number of As
        k = len(self.Bset_)

        # get hierarchy sizes
        b = [len(Bset_i) for Bset_i in self.Bset_]
        # get Bs
        B = [D[:,Bset_i] for Bset_i in self.Bset_]

        #flag type
        m = np.zeros(len(self.Bset_), dtype=int)

        #define P_0
        P = np.eye(self.n_)

        # output flag
        X = []
        R = []
        for i in range(k):

            #compute C
            C = P @ B[i]

            # if all of Bi is in the nullspace of Ci then set it to 0
            C[np.isclose(C, 0, atol=self.zero_tol_)] = 0
            if np.all(C == 0) and len(self.flag_type_) == 0:
                m[i] = 0

            #otherwise continue with flagrep
            else:
                # get basis
                if len(self.flag_type_) == 0:
                    # unknown flag type
                    X.append(self.get_basis(C))
                    m[i] = X[-1].shape[1]
                else:
                    # known flag type
                    if i == 0:
                        m[i] = self.flag_type_[i]
                    else:
                        m[i] = self.flag_type_[i]-self.flag_type_[i-1]
                    X.append(self.get_basis(C, n_vecs = m[i]))

                # get Ri
                Bs = np.hstack([np.zeros((self.n_, b[j])) for j in range(i)]+[B[j] for j in range(i,k)])
                R.append(Bs.T @ P.T @ X[-1])

                # compute projection
                if i < k-1:
                    P = (np.eye(self.n_) - X[-1] @ X[-1].T) @ P
                
                

        # translate to stiefel manifold representative n x n_k
        X = np.hstack(X)
        if X.shape[1] > self.n_:
            print(f'error {np.cumsum(m).astype(int)}')
            X = X[:,:self.n_]

        # compute the flag type (n_1,n_2,...,n_k)
        if len(self.flag_type_) == 0:
            m = m[m != 0] # remove 0s
            self.flag_type_ = np.cumsum(m).astype(int)

        # stack R
        R = np.hstack(R).T

        return X, R

    def inverse_transform(self, X, R):
        """
        Apply the inverse transformation to the data.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            The transformed data to inverse transform.

        Returns:
        X_original: array-like, shape (n_samples, n_features)
            Data in its original form before transformation.
        """

        Dhat = X @ R

        return Dhat

    def objective_value(self, X: np.array, D: np.array = np.empty([]), fl_type: list = []) -> float:
        if len(fl_type) == 0:
            fl_type = self.flag_type_
        
        if len(D) == 0:
            D = self.D_
        else:
            self.n_ = D.shape[0]

        obj_val = 0
        P = np.eye(self.n_)
        # loop through flag
        for i in range(len(self.Bset_)):
            
            if i < len(self.flag_type_):
                n_i = self.flag_type_[i]
            else:
                print('number of subspaces in flag shorter than number feature sets')
                print('... estimating reconstruction using final part of flag')

            if i == 0:
                n_im1 = 0
            else:
                n_im1 = self.flag_type_[i-1]

            Xi = X[:,n_im1: n_i]


            Bset_i = self.Bset_[i]
            obj_val += np.linalg.norm(P @ D[:,Bset_i] - Xi @ Xi.T @ P @ D[:,Bset_i])**2

            
            P = (np.eye(self.n_) - Xi @ Xi.T) @ P
            
            
        
        return obj_val


    def truncate_svd(self, C: np.array, n_vecs: int = 0) -> np.array:
        U,S,_ = np.linalg.svd(C, full_matrices=False)
        

        if n_vecs > 0:
            U = U[:,:n_vecs]
        else:
            S = S/S.max()

            # idea from here: https://inria.hal.science/hal-02326070/file/demo.pdf
            # n_vecs = np.sum(np.array([np.linalg.norm(S[-k:]) for k in range(len(S))]) >= 1e-1)
            # U = U[:,:n_vecs]


            # eigengap thresholding
            # S_gaps = -np.diff(S)
            # for s_gap in S_gaps:
            #     if s_gap >= 1e-3:
            #         n_vecs +=1
            # U = U[:,:n_vecs]

            # pca-inspired
            nnz_ids = ~np.isclose(S, 0, atol=self.zero_tol_)
            U = U[:,nnz_ids]
            S = S[nnz_ids]
            s_prop = np.cumsum(S**2)/np.sum(S**2)
            n_vecs = np.sum(s_prop<=(self.eps_rank_+ self.zero_tol_))
            U = U[:,:n_vecs]

        if self.plot_eigs_:
            plt.figure()
            plt.plot(S)
            if n_vecs > 0:
                plt.vlines(x = n_vecs, ymin =0, ymax = S.max(), colors = 'tab:red', ls = 'dashed')

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
    
    def irls_svd(self, C: np.array, n_vecs: int = 0) -> np.array:
        
        U0 = self.truncate_svd(C, n_vecs)
        ii=0
        err = 1

        pi = C.shape[1]
       
        while ii < 100 and err > 1e-10:
            C_weighted = []
            weights = np.zeros(pi)
            for i in range(pi):
                c = C[:,[i]]
                sin_sq = c.T @ c - c.T @ U0 @ U0.T @ c
                sin_sq = np.max(np.array([sin_sq[0,0], 1e-8]))
                weights[i] = sin_sq**(-1/4)
            
            C_weighted =  C @ np.diag(weights)



            U1 = self.truncate_svd(C_weighted, n_vecs)
            err = np.abs(np.linalg.norm(U1 @ U1.T - U0 @ U0.T))
            U0 = U1.copy()
            ii+=1


        return U0

    def get_basis(self, C, n_vecs: int = 0):
        if self.solver_ == 'svd':
            if len(self.flag_type_) > 0:
                U = self.truncate_svd(C, n_vecs = n_vecs)
            else:
                U = self.truncate_svd(C)   

        elif self.solver_ == 'irls svd':
            if len(self.flag_type_) > 0:
                U = self.irls_svd(C, n_vecs = n_vecs)
            else:
                U = self.irls_svd(C)   

        elif self.solver_ == 'qr':
            if len(self.flag_type_) > 0:
                U = self.truncate_qr(C, n_vecs = n_vecs)
            else:
                U = self.truncate_qr(C)
        
        else:
            raise ValueError('Solver must be either qr or svd or irls svd')
        
        return U


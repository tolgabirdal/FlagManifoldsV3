import numpy as np
import scipy
from matplotlib import pyplot as plt
from FlagRep import FlagRep
from GGD import ggd

class RobustFlagRep(FlagRep):
    def __init__(self, Aset: list = [], flag_type: list = [], 
                 eps_rank: float = 1, zero_tol: float = 1e-10,
                 solver = 'irls svd', plot_eigs = False):

        super().__init__(Aset, flag_type, eps_rank, zero_tol, solver, plot_eigs)
        


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

        # output flag
        X = []


        # get the number of As
        k = len(self.Aset_)

        Rs = {}

        # for feature indices
        Bset = []
        B = []
        for i in range(k):
            if i == 0:
                Bset.append(self.Aset_[0])
            else:
                Bset.append(np.setdiff1d(self.Aset_[i],self.Aset_[i-1]))
            B.append(self.D_[:,Bset[i]])

        

        #weights
        w = []
        m = np.zeros(len(self.Aset_))
        
        Ps = []
        P = np.eye(self.n_)
        Ps.append(P)
        

        for i in range(k):

           
            C = P @ B[i]
            C[np.isclose(C, 0, atol=self.zero_tol_)] = 0
            if np.all(C == 0) and len(self.flag_type_) == 0:
                m[j] = 0
            else:
                if len(self.flag_type_) > 0:
                    if i == 0:
                        i0 = 0
                    else:
                        i0 = self.flag_type_[i-1]
                    i1 = self.flag_type_[i]
                    U, wi = self.get_basis(C, n_vecs = i1-i0)
                else:
                    U, wi = self.get_basis(C)

                w.append(wi)

                X.append(U)
                
                for j in range(i,len(self.Aset_)):
                    if i == j:
                        Rs[(i,j)] = X[i].T @ P @ B[j] @ np.diag(w[j])
                    else:
                        Rs[(i,j)] = X[i].T @ P @ B[j]
            
                if i < k-1:
                    P = (np.eye(self.n_) - X[-1] @ X[-1].T) @ P

                if len(self.flag_type_) == 0:
                    m[j] = X[-1].shape[1]
                
                

        # translate to stiefel manifold representative n x n_k
        X = np.hstack(X)
        if X.shape[1] > self.n_:
            print(f'error {np.cumsum(m).astype(int)}')
            X = X[:,:self.n_]

        # compute the flag type (n_1,n_2,...,n_k)
        if len(self.flag_type_) == 0:
            m = m[m != 0] # remove 0s
            self.flag_type_ = np.cumsum(m).astype(int)
        

        n_k = X.shape[1]
        R = np.zeros((n_k,self.p_))
            
        for i in range(k):
            if i == 0:
                i0 = 0
            else:
                i0 = self.flag_type_[i-1]
            i1 = self.flag_type_[i]

            Xi = X[:,i0:i1]


            for j in range(i,len(self.Aset_)):
                if j == 0:
                    j0 = 0
                else:
                    j0 = len(self.Aset_[j-1])
                j1 = len(self.Aset_[j])

                # if i == j:
                #     Rij = Xi.T @ Ps[i] @ B[j] @ np.diag(w[j])
                # else:
                #     Rij = Xi.T @ Ps[i] @ B[j]

                # Rs = []
                # for l in range(len(Bset[j])):
                #     if i == j:
                #         Rs.append(w[j][l]* Xi.T @ Ps[i] @ B[j][:,[l]])
                #     else:
                #         Rs.append(Xi.T @ Ps[i] @ B[j][:,[l]])

                R[i0:i1,j0:j1] = Rs[(i,j)] #@ np.diag(w[j])

        return X, R



 
    def irls_svd(self, C: np.array, n_vecs: int = 0) -> np.array:
        
        U0 = self.truncate_svd(C, n_vecs)
        ii=0
        err = 1

        pi = C.shape[1]
       
        while ii < 50 and err > 1e-10:
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


        return U0, weights

    def get_basis(self, C, n_vecs: int = 0):

        if self.solver_ == 'irls svd':
            if len(self.flag_type_) > 0:
                U, w = self.irls_svd(C, n_vecs = n_vecs)
            else:
                U, w  = self.irls_svd(C)  

        # elif self.solver_ == 'ggd':
        #     if len(self.flag_type_) > 0:
        #         U = ggd(C.T, s=1e-1, maxiter=100, d=n_vecs, opt=0)
        #     else:
        #         ValueError('must provide flag type for solver = ggd')     
        
        else:
            raise ValueError('Solver must be irls svd')
        
        return U, w


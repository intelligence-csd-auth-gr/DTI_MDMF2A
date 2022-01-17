import numpy as np
from sklearn.neighbors import NearestNeighbors
from base.transbase import TransductiveModelBase

class NRLMF_TRANS(TransductiveModelBase):
    """
    Implementation of NRLMF: 
    Yong L , Min W , Chunyan M , et al. Neighborhood Regularized Logistic Matrix Factorization for Drug-Target Interaction Prediction[J]. PLOS Computational Biology, 2016, 12(2):e1004760.
    """
    
    def __init__(self, cfix=5, K1=5, K2=5, num_factors=10, theta=1.0, lambda_d=0.625, lambda_t=0.625, alpha=0.1, beta=0.1, max_iter=100, seed=0):
        self.cfix = cfix # importance level for positive observations
        self.K1 = K1
        self.K2 = K2
        self.num_factors = num_factors # letent feature of drug and target
        self.theta = theta  # learning rate
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.seed = seed
        
        self.copyable_attrs = ['cfix','K1','K2','num_factors','theta','lambda_d','lambda_t','alpha','beta','max_iter','seed']    
        """
        num_factors = [50,100]
        lambda_t = lambda_d = [2**-5, ..., 2**1] 
        alpha = [2**-5, ..., 2**2]
        beta = [2**-5, ..., 2**0]
        """
    #----------------------------------------------------------------------------------------
        
    def _AGD_optimization(self):
        prng = np.random.RandomState(self.seed)
        self._U = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self._n_drugs, self.num_factors))
        self._V = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self._n_targets, self.num_factors))
            
        dg_sum = np.zeros((self._n_drugs, self._U.shape[1]))
        tg_sum = np.zeros((self._n_targets, self._V.shape[1]))
        last_log = self._log_likelihood()  
        for t in range(self.max_iter):
            dg = self._deriv(True)
            dg_sum += np.square(dg)
            vec_step_size = self.theta / np.sqrt(dg_sum) 
            self._U += vec_step_size * dg
            
            tg = self._deriv(False)
            tg_sum += np.square(tg)
            vec_step_size = self.theta / np.sqrt(tg_sum)
            self._V += vec_step_size * tg
            
            curr_log = self._log_likelihood()
            delta_log = (curr_log-last_log)/abs(last_log)
            if abs(delta_log) < 1e-5:
                break
            last_log = curr_log
            # print(t,'\t',curr_log)
    #----------------------------------------------------------------------------------------
            
    def _deriv(self, drug):
        """compute -1*Eq.13 """
        if drug:
            vec_deriv = np.dot(self._intMat, self._V)  # YV  the 3 term in Eq.13
        else:
            vec_deriv = np.dot(self._intMat.T, self._U)
        A = np.dot(self._U, self._V.T)
        A = np.exp(A)
        A /= (A + self._ones)  # A is the P (predictions)
        A = self._intMat1 * A  # A is P+(c-1)(Y o P)
        if drug:
            vec_deriv -= np.dot(A, self._V)  # the 1,2 terms in Eq.13
            vec_deriv -= self.lambda_d*self._U+self.alpha*np.dot(self._DL, self._U) # last term in Eq.13
        else:
            vec_deriv -= np.dot(A.T, self._U)
            vec_deriv -= self.lambda_t*self._V+self.beta*np.dot(self._TL, self._V)
        return vec_deriv
    #----------------------------------------------------------------------------------------

    def _log_likelihood(self):
        """compute -1*Eq.12 """
        loglik = 0
        A = np.dot(self._U, self._V.T)  #A[i,j]=np.dot(u_i,v_j)
        B = A * self._intMat
        loglik += np.sum(B) # term 1 in Eq.6
        A = np.exp(A)
        A += self._ones
        A = np.log(A)
        A = self._intMat1 * A # intMat1: 1+c*y_ij-y_ij  Eq.6
        loglik -= np.sum(A) # term 2 in Eq.6
        loglik -= 0.5 * self.lambda_d * np.sum(np.square(self._U))+0.5 * self.lambda_t * np.sum(np.square(self._V)) 
        # term 3,4 in Eq.6
        loglik -= 0.5 * self.alpha * np.sum(np.diag((np.dot(self._U.T, self._DL)).dot(self._U))) # Eq.10
        loglik -= 0.5 * self.beta * np.sum(np.diag((np.dot(self._V.T, self._TL)).dot(self._V)))  # Eq.11
        return loglik # the last 3 lines are the last 2 terms in Eq.12
    #----------------------------------------------------------------------------------------

    def _construct_neighborhood(self, drugMat, targetMat):
        dsMat = drugMat - np.diag(np.diag(drugMat))  # dsMat is the drugMat which sets the diagonal elements to 0 
        tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K1 > 0:
            S1 = self._get_nearest_neighbors(dsMat, self.K1)  # S1 is sparsified durgMat A
            self._DL = self._laplacian_matrix(S1)                   # L^d
            S2 = self._get_nearest_neighbors(tsMat, self.K1)  # S2 is sparsified durgMat B
            self._TL = self._laplacian_matrix(S2)                   # L^t
        else:
            self._DL = self._laplacian_matrix(dsMat)
            self._TL = self._laplacian_matrix(tsMat)
    #----------------------------------------------------------------------------------------

    def _laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        L = 0.5*(np.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L
    #----------------------------------------------------------------------------------------

    def _get_nearest_neighbors(self, S, size=5):
        """ Eq.8, Eq.9, the S is the similarity matrix whose diagonal elements are 0"""
        m, n = S.shape
        X = np.zeros((m, n))
        neigh = NearestNeighbors(n_neighbors=size, metric='precomputed')
        neigh.fit(np.zeros((m,n)))
        knn_indices = neigh.kneighbors(1-S, return_distance=False) # 1-S is the distance matrix whose diagonal elements are 0
        for i in range(m):
            ii = knn_indices[i]
            X[i, ii] = S[i, ii]
        return X
    #----------------------------------------------------------------------------------------

    def fit(self, intMat, drugMat, targetMat, test_indices, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, test_indices, cvs)
        self.lambda_t = self.lambda_d # ensure self.lambda_t = self.lambda_d 
            
        self._ones = np.ones((self._n_drugs, self._n_targets))
        self._intMat = self.cfix*intMat  # positive interations weighted as cifx, 
        self._intMat1 = (self.cfix-1)*intMat + self._ones  # intMat1=(c-1)Y+1  {1+cy_ij-y_ij  Eq.6}
        self._construct_neighborhood(drugMat, targetMat)
        self._AGD_optimization()
        
        # make prediction of whole imtMat
        if self._cvs == 1: 
            U_te = self._U
            V_te = self._V
        elif self._cvs == 2:
            test_d = test_indices # test drug indices 
            U_te = self._get_test_U(self._U, drugMat, test_d)
            V_te = self._V
        elif self._cvs == 3:
            U_te = self._U
            test_t = test_indices
            V_te = self._get_test_U(self._V, targetMat, test_t)
        elif self._cvs == 4: 
            test_d,test_t = test_indices
            U_te = self._get_test_U(self._U, drugMat, test_d)
            V_te = self._get_test_U(self._V, targetMat, test_t)
            
        scores = U_te@V_te.T
        exp_s = np.exp(scores)
        scores =exp_s/(1+exp_s)
        # get the prediction of test pairs
        S_te = self._get_test_scores(scores)
        S_te[S_te==np.inf] = 0.0
        S_te[np.isnan(S_te)] = 0.0
        return S_te
    #----------------------------------------------------------------------------------------
    
    def _get_test_U(self, U, drugMat, test_d):
        nd = drugMat.shape[0]
        all_d = np.arange(nd, dtype=int)
        train_d = np.setdiff1d(all_d,test_d) # training drug indices 
        D_te = drugMat[:,train_d] # columns for training drugs
        D_te = D_te[test_d,:] # # rows for training drugs
        neigh_d = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
        neigh_d.fit(np.zeros((len(train_d),len(train_d))))
        knn_d = neigh_d.kneighbors(1-D_te, return_distance=False) # only contain knn of test drugs
        
        U_te = np.copy(U)
        U_tr = U[train_d,:] # only latent features of training drugs
        for i in range(test_d.size):
            d = test_d[i]
            ii = knn_d[i]
            z = np.sum(D_te[i, ii])
            if z == 0:
                z = 1
            U_te[d,:]= np.dot(D_te[i, ii], U_tr[ii, :])/z
        return U_te
    #----------------------------------------------------------------------------------------
    
    def __str__(self):
        return "Model: NRLMF_TRANS, c:%s, K1:%s, K2:%s, r:%s, lambda_d:%s, lambda_t:%s, alpha:%s, beta:%s, theta:%s, max_iter:%s" % (self.cfix, self.K1, self.K2, self.num_factors, self.lambda_d, self.lambda_t, self.alpha, self.beta, self.theta, self.max_iter)


    
        
    

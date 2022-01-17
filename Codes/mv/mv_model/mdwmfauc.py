import numpy as np
import copy

from sklearn.neighbors import NearestNeighbors
from base.mvinbase import MultiViewInductiveModelBase
# from base.tensor_3d import *
from scipy import linalg #  linalg.khatri_rao : Khatri-Rao product (column-wise Kronecker product)
# from scipy.optimize import minimize,Bounds
# from sklearn.linear_model import Ridge
from scipy.linalg import block_diag
from mv.com_sim.combine_sims import Combine_Sims_Ave

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc, average_precision_score


class MDMFAUC(MultiViewInductiveModelBase):
    """learning a joint/global feature matrix that capture infromaition from multiple similairties 
    """
    
    def __init__(self, K=5, K2=5, metric=0, Ts=np.arange(0,1.1,0.1), cs_model=Combine_Sims_Ave(), num_factors=50, theta=0.1, lambda_d=0.25, lambda_t=0.25, lambda_r=0.25, lambda_M=0.005, n_wz=5, num_neg=1, max_iter=100, seed=0, sfun=0, is_comLoss=0, isAGD=True, is_sparseM = False):
        self.K = K # used for sparsying similarity matrix and computing latent features of new drugs/targets
        self.K2 = K2 # K2 used for compute test drug/target latent feature
        self.Ts = Ts # candidates decay coefficients
        self.metric= metric # metric to choose T, 0: aupr; 1: aupr+auc
        
        self.cs_model = cs_model
        self.num_factors = num_factors # letent feature of drug and target
        self.theta = theta  # learning rate
        self.lambda_d = lambda_d  # coefficient of graph based regularization of U
        self.lambda_t = lambda_t  # coefficient of graph based regularization of V
        self.lambda_r = lambda_r  # coefficient of ||U||_F^2+||V||_F^2 regularization
        self.lambda_M = lambda_M  # coefficient of ||M-QQ'||_F^2, factorizing M
        self.max_iter = max_iter
        self.seed = seed
        
        
        self.n_wz = n_wz # the context window size
        self.num_neg = num_neg # the number of negative samples, 1 is best, 5 leads te M too sparse
        
        self.sfun = sfun # the convex surrogate loss function 
        """(0: square loss; 1: square hinge loss; 2: logistic loss)"""
        self.is_comLoss = is_comLoss # if compute loss for each iteration or not (0: do not compute; 1: compute all Loss; 2 compute AUC only)
        self.isAGD = isAGD
        self.is_sparseM = is_sparseM # if use sparse similarity to compute M matrix
        
        self.copyable_attrs = ['K','K2','metric','Ts','cs_model','num_factors','theta', 'lambda_d','lambda_t','lambda_r','lambda_M','n_wz','num_neg','max_iter', 'seed','sfun','is_comLoss','isAGD','is_sparseM']
    #----------------------------------------------------------------------------------------

    def fit(self, intMat, drugMats, targetMats, cvs=2): 
        self._check_fit_data(intMat, drugMats, targetMats, cvs)
        self._Y = intMat
        self._init_f_loss_and_deriv()
        
        Sds, Sts = self._construct_neighborhood(drugMats, targetMats) # 
        
        _,self._ad = self.cs_model.combine(drugMats, intMat)
        _,self._at = self.cs_model.combine(targetMats, intMat.T)
        # print(self._ad, '\t', self._at)
        
        if self.is_sparseM:
            self._M_mat = self.compute_M_matrix(self._Y,  Sds, Sts) 
        else:
            self._M_mat = self.compute_M_matrix(self._Y,  drugMats, targetMats) 
        self._O = np.ones(self._M_mat.shape)
        # self._O -= np.eye(self._O.shape[0])
        # self._O[self._M_mat == 0] = 0
        
        self._AGD_optimization(drugMats, targetMats) 
        self._get_optimal_T(drugMats, targetMats)
    #----------------------------------------------------------------------------------------
    
    def compute_M_matrix(self, Y, Sds, Sts):
        M_mats = np.zeros((self._n_dsims*self._n_tsims, self._n_drugs+self._n_targets, self._n_drugs+self._n_targets))
        wm = np.outer(self._ad, self._at).flatten() # the weight of M_mat
        M_ind = 0
        for Sd in Sds:
            for St in Sts:
                A1 = np.concatenate((Sd, Y), axis=1)
                A2 = np.concatenate((Y.T, St), axis=1)
                A = np.concatenate((A1, A2), axis=0)
                
                volG = A.sum()
                L1 = np.diag(1/A.sum(axis=1)) # Λ^-1
                L1[np.isinf(L1)] = 0 # set inf numbers if A.sum contains zeros
                P = L1@A # Λ^-1A
                M_mat1 = np.zeros(A.shape)
                Pi = P # (Λ^-1A)^i
                for i in range(self.n_wz):
                    M_mat1 += Pi
                    Pi = Pi@P
                M_mat1 = volG*(M_mat1/self.n_wz/self.num_neg)@L1
                M_mat1[M_mat1<1] = 1 # all values in M_mat >= 0
                M_mat = np.log(M_mat1) 
                M_mats[M_ind] = M_mat
                M_ind += 1
                    
        M_mat = np.average(M_mats, axis=0, weights=wm)
        return M_mat  
    #----------------------------------------------------------------------------------------       
    
    
    def _get_M_mat_blocks(self):
        M_Sd = self._M_mat[:self._n_drugs, :self._n_drugs] 
        M_St = self._M_mat[self._n_drugs:, self._n_drugs:]
        M_Y = self._M_mat[:self._n_drugs, self._n_drugs:]
        return M_Sd, M_St, M_Y
    #---------------------------------------------------------------------------------------- 
    
    def _AGD_optimization(self, drugMats, targetMats):
        self._prng = np.random.RandomState(self.seed)
        self._U = np.sqrt(1/float(self.num_factors))*self._prng.normal(size=(self._n_drugs, self.num_factors))
        self._V = np.sqrt(1/float(self.num_factors))*self._prng.normal(size=(self._n_targets, self.num_factors))
        
        M_Sd, M_St, M_Y = self._get_M_mat_blocks()
        
        self._idx1 = np.where(self._Y==1) # (array([0, 1], dtype=int64), array([2, 0], dtype=int64))
        self._idx0 = np.where(self._Y==0) # (array([0, 0, 1, 1], dtype=int64), array([0, 1, 1, 2], dtype=int64))
        
        if self.is_comLoss == 1:
            # the 1d index is for compute AUC loss
            idx1_1d = np.ravel_multi_index(self._idx1,self._Y.shape) # array([0,1,4,5])
            idx0_1d = np.ravel_multi_index(self._idx0,self._Y.shape) # array([2,3])
            i1_ = np.tile(idx1_1d,len(idx0_1d))  # i1_ = array([0,1,4,5,0,1,4,5])
            i0_ = np.repeat(idx0_1d,len(idx1_1d)) # i0_ = array([2,2,2,2,3,3,3,3])
            pair_idx_1d = (i0_,i1_) # index of negative and positive paris 
        
        if self.isAGD:
            du_sum = np.zeros(self._U.shape)
            dv_sum = np.zeros(self._V.shape)
        else:
            current_theta = self.theta
        
        
        if self.is_comLoss == 1:
            last_loss, lauc, r_r, r_d, r_t , r_M, auc_val= self._compute_loss(pair_idx_1d, self._f_loss) 
            print('\t',round(last_loss,6),'\t', round(lauc,6),'\t',round(r_r,6),'\t',round(r_d,6),'\t',round(r_t,6),'\t',round(r_M,6),'\t',round(auc_val,6))
        
        for iteration in range(self.max_iter):            
            idx1_s, idx0_s = self._sampling_pairs()
            
            # update U
            Y_pred = self._get_prediction_trainset()
            
            deriv_U_M = (self._U@self._U.T - M_Sd)@self._U
            deriv_U_M += (self._U@self._V.T - M_Y)@self._V
            deriv_U_M *= 2*self.lambda_M

            deriv_U = self.lambda_r*self._U #/self._n_drugs
            for i in range(self._n_dsims):
                deriv_U += self.lambda_d*self._ad[i]*self._DLs[i]@self._U #/self._n_drugs
            du = self._deriv_AUCLoss(self._U, self._V, idx1_s, idx0_s, Y_pred, [0,1], self._f_loss_deriv)
            du = du + deriv_U + deriv_U_M
            if self.isAGD:
                du_sum += np.square(du)
                vec_step_size_d = self.theta / np.sqrt(du_sum) 
                self._U -= vec_step_size_d * du
            else:
                self._U -= current_theta* du
            
            
            # update V
            Y_pred = self._get_prediction_trainset()
            
            deriv_V_M = (self._V@self._V.T - M_St)@self._V
            deriv_V_M += (self._V@self._U.T - M_Y.T)@self._U
            deriv_V_M *= 2*self.lambda_M
            
            # Q = np.concatenate((self._U, self._V), axis=0)
            # deriv_Q = 2*self.lambda_M*self._O*(Q@Q.T-self._M_mat)@Q
            # deriv_V_M = deriv_Q[self._n_drugs:]
            
            deriv_V = self.lambda_r*self._V #/self._n_targets
            for i in range(self._n_tsims):
                deriv_V += self.lambda_t*self._at[i]*self._TLs[i]@self._V #/self._n_targets
            dv = self._deriv_AUCLoss(self._V, self._U, idx1_s, idx0_s, Y_pred.T, [1,0], self._f_loss_deriv)
            dv = dv + deriv_V + deriv_V_M
            if self.isAGD:
                dv_sum += np.square(dv)
                vec_step_size = self.theta / np.sqrt(dv_sum)
                self._V -= vec_step_size * dv
            else:
                self._V -= current_theta * dv

            
            if self.is_comLoss == 1:
                curr_loss, lauc, r_r, r_d, r_t , r_M, auc_val= self._compute_loss(pair_idx_1d, self._f_loss) 
                if np.isnan(curr_loss) or np.isinf(curr_loss):
                    # The model fails to be trained, the coefficient of regularizatin terms are too large
                    self._U = np.zeros((self._n_drugs, self.num_factors))
                    self._V = np.zeros((self._n_targets, self.num_factors))
                    break
                delta_loss = (curr_loss-last_loss)/abs(last_loss)
                print(iteration,'\t',round(curr_loss,6),'\t', round(lauc,6),'\t',round(r_r,6),'\t',round(r_d,6),'\t',round(r_t,6),'\t',round(r_M,6),'\t',round(auc_val,6))
                if self.isAGD:
                    if abs(delta_loss) < 1e-6:
                        break
                else:
                    if delta_loss>0: # abs(delta_loss) < 1e-5: 
                        current_theta *= 0.9
                    if abs(delta_loss) < 1e-6:
                        break
                last_loss = curr_loss
            elif self.is_comLoss ==2:
                Y_pred = self._get_prediction_trainset()
                y_pred = Y_pred.flatten()
                auc_val = self._compute_auc(self._Y.flatten(), y_pred)
                print(iteration,'\t','\t',round(auc_val,6))
            # print(iteration)
    #----------------------------------------------------------------------------------------
    
    
    def _init_f_loss_and_deriv(self):
        if self.sfun == 0:
            self._f_loss = self.fSL
            self._f_loss_deriv = self.fSL_deriv
        elif self.sfun == 1:
            self._f_loss = self.fSHL
            self._f_loss_deriv = self.fSHL_deriv
        elif self.sfun == 2:
            self._f_loss = self.fLL
            self._f_loss_deriv = self.fLL_deriv
        else:
            self._f_loss = None
            self._f_loss_deriv = None          
    #----------------------------------------------------------------------------------------  
    
    def fSL(self, x):
        """ square loss function"""
        return np.power(1-x,2)/2
    #----------------------------------------------------------------------------------------
    
    def fSL_deriv(self, x):
        """ derivative of square loss function"""
        return x-1
    #---------------------------------------------------------------------------------------- 
    
    def fSHL(self, x):
        """ square hinge loss function"""
        y = 1-x
        y[y<0] = 0
        return np.power(y,2)         
    #----------------------------------------------------------------------------------------
    
    def fSHL_deriv(self, x):
        """ derivative of square hinge loss function"""
        y=x-1
        if isinstance(y, np.ndarray):    
            y[y>0] = 0
        else:
            y = min(y,0)
        return y
    #---------------------------------------------------------------------------------------- 
    
    def fLL(self, x):
        """ logistic loss function"""
        return np.log(1+np.exp(-x))
    #----------------------------------------------------------------------------------------
    
    def fLL_deriv(self, x):
        """ derivative of logistic loss function"""
        return -1/(1+np.exp(x))
    #----------------------------------------------------------------------------------------  
    
    def _get_prediction_trainset(self):
        return self._U@self._V.T
    #----------------------------------------------------------------------------------------  
    
    
    def _compute_loss(self, pair_idx_1d, func_loss):
        Y_pred = self._get_prediction_trainset()
        
        y_pred = Y_pred.flatten() # transform Y_pred to 1d vector
        # print(np.max(y_pred), '\t', np.min(y_pred))
        diff = y_pred[pair_idx_1d[1]] - y_pred[pair_idx_1d[0]] # the u_i*v_j - u_h*v_l for all pairs where Y_ij=1 and Y_hl=0
        diff1 = func_loss(diff)
        lauc = diff1.sum() #diff1.mean()

        r_r = 0.5*self.lambda_r*np.square(np.linalg.norm(self._U,'fro'))#/self._n_drugs 
        r_r += 0.5*self.lambda_r*np.square(np.linalg.norm(self._V,'fro'))#/self._n_targets
        
        r_d = 0
        for i in range(self._n_dsims):
            r_d += self._ad[i]*np.trace(self._U.T@self._DLs[i]@self._U)#/self._n_drugs
        r_d *= 0.5*self.lambda_d
        r_t = 0
        for i in range(self._n_tsims):
            r_t += self._at[i]*np.trace(self._V.T@self._TLs[i]@self._V)#/self._n_targets
        r_t *= 0.5*self.lambda_t
        # print(r_ds, '\t', r_ts )
        
        Q = np.concatenate((self._U, self._V), axis=0)
        r_M = 0.5*self.lambda_M*np.square(np.linalg.norm(self._O*(self._M_mat-Q@Q.T),'fro'))
        
        loss = lauc + r_r + r_d + r_t + r_M
        
        auc_val = self._compute_auc(self._Y.flatten(), y_pred)

        return loss, lauc, r_r, r_d, r_t, r_M, auc_val
    #----------------------------------------------------------------------------------------
    
    
    def _compute_auc(self, labels_1d, scores_1d):
        fpr, tpr, thr = roc_curve(labels_1d, scores_1d)
        auc_val = auc(fpr, tpr)
        return auc_val
    #----------------------------------------------------------------------------------------
    def _compute_aupr(self, labels_1d, scores_1d):
        aupr_val = average_precision_score(labels_1d,scores_1d)
        if np.isnan(aupr_val):
            aupr_val=0.0
        return aupr_val    
    #----------------------------------------------------------------------------------------
    
    
    def _sampling_pairs(self):
        """ random sampling pairs"""
        n1 = self._idx1[0].size
        n0 = self._idx0[0].size
        n = n0+n1 
        index1 = self._prng.choice(np.arange(n1), n)
        idx1_s = (self._idx1[0][index1], self._idx1[1][index1])
        
        index0 = self._prng.choice(np.arange(n0), n)
        idx0_s = (self._idx0[0][index0], self._idx0[1][index0])
        
        return idx1_s, idx0_s 
    #----------------------------------------------------------------------------------------    

    def _deriv_AUCLoss(self, U, V, idx1_s, idx0_s, Y_pred, idx_flag, func_loss_deriv):
        """compute the derivatives of AUC loss w.r.t U or V, idx_flag = [0,1] for U and [1,0] for V        
        """
        n,m = Y_pred.shape
        deriv_U = np.zeros(U.shape)
        
        if idx_flag[0] == 1:  #deriv of V, exchange the tuple elements in idx1_s, idx0_s
            i1, j1 = idx1_s
            idx1_s = j1, i1
            i0, j0 = idx0_s
            idx0_s = j0, i0
        
        Y1 = Y_pred[idx1_s]
        Y0 = Y_pred[idx0_s]
        diff = Y1-Y0
                
        du1 = func_loss_deriv(diff)[:,None]*V[idx1_s[1]]
        du0 = -func_loss_deriv(diff)[:,None]*V[idx0_s[1]] 
        
        deriv_U1 = np.array([du1[idx1_s[0]==i].sum(axis=0) for i in range(n)]) # 2d array shape=(n, num_factors), deriv_U1[index] is the sum of rows of du1 whose corresponding i=index    
        deriv_U0 = np.array([du0[idx0_s[0]==i].sum(axis=0) for i in range(n)])
        deriv_U = deriv_U1+deriv_U0
        
        # this part of code has same function with above three lines and similar running time 
        # for i1 in range(n):
        #     du = du1[idx1_s[0]==i1]
        #     sdu = np.sum(du, axis=0)
        #     deriv_U[i1] += sdu
            
        #     du = du0[idx0_s[0]==i1]
        #     sdu = np.sum(du, axis=0)
        #     deriv_U[i1] += sdu
            
        return deriv_U
    #----------------------------------------------------------------------------------------
    

    
    def _construct_neighborhood(self, drugMats, targetMats):
        # construct the laplocian matrices and Ms tensor
        self._DLs = np.zeros(drugMats.shape)
        self._TLs = np.zeros(targetMats.shape)
        Sds = np.zeros(drugMats.shape)
        Sts = np.zeros(targetMats.shape)
        
        for i in range(self._n_dsims):
            dsMat = drugMats[i] - np.diag(np.diag(drugMats[i]))  # dsMat is the drugMat which sets the diagonal elements to 0 
            if self.K > 0:
                Sds[i], _  = self._get_nearest_neighbors(dsMat, self.K)  # S1 is sparsified durgMat A
                self._DLs[i] = self._laplacian_matrix(Sds[i])                   # L^d
            else:
                self._DLs[i] = self._laplacian_matrix(dsMat)
                
        for i in range(self._n_tsims):
            tsMat = targetMats[i] - np.diag(np.diag(targetMats[i]))
            if self.K > 0:
                Sts[i], _ = self._get_nearest_neighbors(tsMat, self.K)  # S2 is sparsified durgMat B
                self._TLs[i] = self._laplacian_matrix(Sts[i])                  # L^t
            else:
                self._TLs[i] = self._laplacian_matrix(tsMat)
        return Sds, Sts
    #----------------------------------------------------------------------------------------
    
    # def _laplacian_matrix(self, S):
    #     """ Change to normalized laplacian"""
    #     x = np.sum(S, axis=0)
    #     y = np.sum(S, axis=1)
    #     xy = x+y
    #     L = 0.5*(np.diag(xy) - (S+S.T))  # neighborhood regularization matrix
    
    #     dxy = np.power(xy,-0.5) # d^-0.5
    #     dxy[dxy == np.inf] = 0
    #     Dxy = np.diag(dxy)
    #     L1 = Dxy@L@Dxy # normalized lapacian matrix 
        
    #     return L1
    # #----------------------------------------------------------------------------------------
        
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
        M = np.ones((m, n))
        neigh = NearestNeighbors(n_neighbors=size, metric='precomputed')
        neigh.fit(np.zeros((m,n)))
        knn_indices = neigh.kneighbors(1-S, return_distance=False) # 1-S is the distance matrix whose diagonal elements are 0
        for i in range(m):
            ii = knn_indices[i]
            X[i, ii] = S[i, ii]
        X = (X + X.T)/2  # !!!!!!!!!
        M[X==0] = 0 # ignore the 0 value similarities
        return X, M
    #----------------------------------------------------------------------------------------
    
    def _combine_sims(self, ad, Ss):
        return np.tensordot(ad,Ss,axes=((0),(0)))
    #----------------------------------------------------------------------------------------    

    def _get_optimal_T(self, drugMats, targetMats):
        drugMat = self._combine_sims(self._ad, drugMats)
        targetMat = self._combine_sims(self._at, targetMats)
        
        Sd = drugMat - np.diag(np.diag(drugMat))  # dsMat is the drugMat which sets the diagonal elements to 0 
        St = targetMat - np.diag(np.diag(targetMat))
        
        if self._cvs == 2 or self._cvs == 4:
            neigh_d = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
            neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            knn_d = neigh_d.kneighbors(1-Sd, return_distance=False)
        if self._cvs == 3 or self._cvs == 4:  
            neigh_t = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
            neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            knn_t = neigh_t.kneighbors(1-St, return_distance=False)
        
        best_value = -1; self._best_T = None
        for T in self.Ts:
            etas = T**np.arange(self.K2)
            if self._cvs == 2: 
                U = np.zeros(self._U.shape)
                for d in range(self._n_drugs):
                    ii = knn_d[d]
                    sd = Sd[d,ii]
                    U[d,:]= etas*sd@self._U[ii, :]/np.sum(sd)
                V = self._V
            elif self._cvs == 3:
                V = np.zeros(self._V.shape)
                for t in range(self._n_targets):
                    jj = knn_t[t]
                    st = St[t,jj]
                    V[t,:]= etas*st@self._V[jj, :]/np.sum(st)   
                U = self._U
            elif self._cvs == 4:
                U = np.zeros(self._U.shape)
                for d in range(self._n_drugs):
                    ii = knn_d[d]
                    sd = Sd[d,ii]
                    U[d,:]= etas*sd@self._U[ii, :]/np.sum(sd)
                V = np.zeros(self._V.shape)
                for t in range(self._n_targets):
                    jj = knn_t[t]
                    st = St[t,jj]
                    V[t,:]= etas*st@self._V[jj, :]/np.sum(st)
            Y_pred = U@V.T
            if self.metric == 0:
                auc = self._compute_auc(self._Y.flatten(), Y_pred.flatten())
                value = auc
            elif self.metric == 1:
                aupr = self._compute_aupr(self._Y.flatten(), Y_pred.flatten())
                auc = self._compute_auc(self._Y.flatten(), Y_pred.flatten())
                value = aupr + auc
            if value > best_value:
                best_value = value
                self._best_T = T
        # print(self._best_T)
    #----------------------------------------------------------------------------------------  


    def predict(self, drugMatTes, targetMatTes):
        self._check_test_data(drugMatTes, targetMatTes)       
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)

        drugMatTe = self._combine_sims(self._ad, drugMatTes)
        targetMatTe = self._combine_sims(self._at, targetMatTes)        
        
        
        drug_dis_te = 1 - drugMatTe
        target_dis_te = 1 - targetMatTe
        if self._cvs == 2 or self._cvs == 4:
            neigh_d = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
            neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            knn_d = neigh_d.kneighbors(drug_dis_te, return_distance=False)
        if self._cvs == 3 or self._cvs == 4:  
            neigh_t = NearestNeighbors(n_neighbors=self.K2, metric='precomputed')
            neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            knn_t = neigh_t.kneighbors(target_dis_te, return_distance=False)

        U_te = np.zeros((self._n_drugs_te,self.num_factors), dtype=float)
        V_te = np.zeros((self._n_targets_te,self.num_factors), dtype=float)
        etas = self._best_T**np.arange(self.K2)
        if self._cvs == 2: 
            for d in range(self._n_drugs_te):
                ii = knn_d[d]
                U_te[d,:]= etas*drugMatTe[d, ii]@self._U[ii, :]/np.sum(drugMatTe[d, ii])
            V_te = self._V
        elif self._cvs == 3:
            for t in range(self._n_targets_te):
                jj = knn_t[t]
                V_te[t,:]= etas*targetMatTe[t, jj]@self._V[jj, :]/np.sum(targetMatTe[t, jj])   
            U_te = self._U
        elif self._cvs == 4:
            for d in range(self._n_drugs_te):
                ii = knn_d[d]
                U_te[d,:]= etas*drugMatTe[d, ii]@self._U[ii, :]/np.sum(drugMatTe[d, ii])
            for t in range(self._n_targets_te):
                jj = knn_t[t]
                V_te[t,:]= etas*targetMatTe[t, jj]@self._V[jj, :]/np.sum(targetMatTe[t, jj])   
        scores = U_te@V_te.T

        # scores[scores==np.inf] = 0.0
        # scores[np.isnan(scores)] = 0.0
        return scores
    #----------------------------------------------------------------------------------------   
#-------------------------------------------------------------------------------------------------------------------

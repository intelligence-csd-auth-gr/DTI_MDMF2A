import numpy as np
import copy

from sklearn.neighbors import NearestNeighbors
from base.mvtransbase import MultiViewTransductiveModelBase
from base.gipbase import GIPSimBase

from model_trans.mf.nrlmf_trans import NRLMF_TRANS
from mv.com_sim.combine_sims import Combine_Sims_Ave
from base.splitdata import split_train_test_set_mv

from sklearn.metrics.pairwise import rbf_kernel

class LCS_SV_TRANS(MultiViewTransductiveModelBase):
    """
    1. linearly combine the multiple similarities to one similairty matrix
    2. train a DTI predicton mehtod using the combined/fused durg and target similairties
    """
    
    def __init__(self, cs_model=Combine_Sims_Ave(), sv_model=NRLMF_TRANS(cfix=5, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0)):
        self.cs_model = cs_model # method to combine multiple similarities to one similarity matrix
        self.sv_model = sv_model # preidiction method using one similarity matrix (single view)
        
        self.copyable_attrs = ['cs_model','sv_model']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMats, targetMats, test_indices, cvs=2):
        self._check_fit_data(intMat, drugMats, targetMats, test_indices, cvs)
        # get weihgts of each input similarities and combined/fused similarities
        D, self.wd = self.cs_model.combine(drugMats, intMat)
        T, self.wt = self.cs_model.combine(targetMats, intMat.T)
        # train DTI prediciton model based on combined similairties
        S_te = self.sv_model.fit(intMat, D, T, test_indices, cvs)
        return S_te
    #----------------------------------------------------------------------------------------
    
    def _get_prediction_trainset(self):
        S = self.sv_model._get_prediction_trainset()
        return S
    #----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------   
    
    
class LCS_SV_GIP_TRANS(LCS_SV_TRANS, GIPSimBase):
    """
    Different with LCS_SV: add GIP similarity
    """
    def fit(self, intMat, drugMats, targetMats, test_indices, cvs=2):
        self._check_fit_data(intMat, drugMats, targetMats, test_indices, cvs)
        
        if self._cvs == 1:
            Yd = intMat
            Yt = intMat.T
        elif self._cvs == 2:
            test_d = test_indices # test drug indices
            all_d = np.arange(self._n_drugs)
            train_d = np.setdiff1d(all_d, test_d)
            Yd = self._impute_Y_test(intMat, drugMats[0], train_d, test_d)
            Yt = intMat.T  
        elif self._cvs == 3:
            Yd = intMat
            test_t = test_indices
            all_t = np.arange(self._n_targets)
            train_t = np.setdiff1d(all_t, test_t)
            Yt = self._impute_Y_test(intMat.T, targetMats[0], train_t, test_t)
        elif self._cvs == 4: 
            test_d,test_t = test_indices
            all_d = np.arange(self._n_drugs)
            train_d = np.setdiff1d(all_d, test_d)
            Yd = self._impute_Y_test(intMat, drugMats[0], train_d, test_d)
            all_t = np.arange(self._n_targets)
            train_t = np.setdiff1d(all_t, test_t)
            Yt = self._impute_Y_test(intMat.T, targetMats[0], train_t, test_t)
        
        GIP_d = self.compute_GIP(Yd)
        Sds = self._add_GIP(drugMats, GIP_d)
        self._n_dsims = self._n_dsims+1
        GIP_t = self.compute_GIP(Yt)
        Sts = self._add_GIP(targetMats, GIP_t)
        self._n_tsims = self._n_tsims+1      
        
        # get weihgts of each input similarities and combined/fused similarities
        D, self.wd = self.cs_model.combine(Sds, intMat)
        T, self.wt = self.cs_model.combine(Sts, intMat.T)
        # train DTI prediciton model based on combined similairties
        S_te = self.sv_model.fit(intMat, D, T, test_indices, cvs)
        return S_te
    #----------------------------------------------------------------------------------------
    
    def _add_GIP(self, Sds, GIP):
        GIP1 = GIP.reshape(1,GIP.shape[0],GIP.shape[1])
        Sds1 = np.concatenate((Sds,GIP1),axis=0)
        return Sds1
    #---------------------------------------------------------------------------------------- 
     
    def _impute_Y_test(self, intMat, Sd, train_d, test_d, k=5):
        Y = np.copy(intMat)
        S = Sd - np.diag(np.diag(Sd))
        S = S[:,train_d] # find kNNs from training drugs
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros((S.shape[1],S.shape[1])))
        knns = neigh.kneighbors(1 - S, return_distance=False)
        
        for d in test_d:
            ii = knns[d]
            sd = S[d,ii]
            Y[d,:] = sd@intMat[ii,:]
            z = np.sum(sd)
            if z>0:
                Y[d,:]/=z                
        return Y
    #----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------

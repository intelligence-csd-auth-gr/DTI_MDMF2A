import numpy as np
import copy

from sklearn.neighbors import NearestNeighbors
from base.mvinbase import MultiViewInductiveModelBase
from base.gipbase import GIPSimBase

from mf.nrlmf import NRLMF
from mv.com_sim.combine_sims import Combine_Sims_Ave



class LCS_SV(MultiViewInductiveModelBase):
    """
    1. linearly combine the multiple similarities to one similairty matrix
    2. train a DTI predicton mehtod using the combined/fused durg and target similairties
    """
    
    def __init__(self, cs_model=Combine_Sims_Ave(), sv_model=NRLMF(cfix=5, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0)):
        self.cs_model = cs_model # method to combine multiple similarities to one similarity matrix
        self.sv_model = sv_model # preidiction method using one similarity matrix (single view)
        
        self.copyable_attrs = ['cs_model','sv_model']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMats, targetMats, cvs=2): 
        self._check_fit_data(intMat, drugMats, targetMats, cvs)
        # get weihgts of each input similarities and combined/fused similarities
        D, self.wd = self.cs_model.combine(drugMats, intMat)
        T, self.wt = self.cs_model.combine(targetMats, intMat.T)
        # train DTI prediciton model based on combined similairties
        self.sv_model.fit(intMat, D, T, cvs)
    #----------------------------------------------------------------------------------------
    
    def predict(self, drugMatTes, targetMatTes):
        self._check_test_data(drugMatTes, targetMatTes)
        # get the combined/fused test similarities
        D_te = np.average(drugMatTes,axis=0,weights=self.wd)
        T_te = np.average(targetMatTes,axis=0,weights=self.wt)
        # predict socres based on combined/fused test similarities
        scores = self.sv_model.predict(D_te, T_te)        
        return scores
    #---------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------- 


class LCS_SV_GIP(LCS_SV, GIPSimBase):
    """
    Different with LCS_SV: add GIP similarity
    """
    
    def fit(self, intMat, drugMats, targetMats, cvs=2): 
        self._check_fit_data(intMat, drugMats, targetMats, cvs)
        
        self._Y = intMat
        
        # add GIP for training set
        self._GIP_d = self.compute_GIP(intMat)
        Sds = self._add_GIP(drugMats, self._GIP_d)
        self._n_dsims = self._n_dsims+1
        self._GIP_t = self.compute_GIP(intMat.T)
        Sts = self._add_GIP(targetMats, self._GIP_t)
        self._n_tsims = self._n_tsims+1
        
        # get weihgts of each input similarities and combined/fused similarities
        D, self.wd = self.cs_model.combine(Sds, intMat)
        T, self.wt = self.cs_model.combine(Sts, intMat.T)
        # train DTI prediciton model based on combined similairties
        self.sv_model.fit(intMat, D, T, cvs)
    #----------------------------------------------------------------------------------------
    
    def predict(self, drugMatTes, targetMatTes):
        GIP_d_te, GIP_t_te = self._compute_GIP_te(drugMatTes[0],targetMatTes[0],self._Y)
        Sds_te = self._add_GIP(drugMatTes, GIP_d_te)
        Sts_te = self._add_GIP(targetMatTes, GIP_t_te)
        
        self._check_test_data(Sds_te, Sts_te)
        
        # get the combined/fused test similarities
        D_te = np.average(Sds_te,axis=0,weights=self.wd)
        T_te = np.average(Sts_te,axis=0,weights=self.wt)
        # predict socres based on combined/fused test similarities
        scores = self.sv_model.predict(D_te, T_te)        
        print(self.wd,'\t',self.wt)
        return scores
    #---------------------------------------------------------------------------------------- 
    
    def _add_GIP(self, Sds, GIP):
        GIP1 = GIP.reshape(1,GIP.shape[0],GIP.shape[1])
        Sds1 = np.concatenate((Sds,GIP1),axis=0)
        return Sds1
    #---------------------------------------------------------------------------------------- 
    

    def _compute_GIP_te(self, Sd_te, St_te, train_Y):    
        self._n_drugs_te = Sd_te.shape[0]
        self._n_targets_te = St_te.shape[0]
        if self._cvs == 2:
            test_Yd = np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
            test_Yd_impute = self._impute_Y_test(train_Y, test_Yd, Sd_te)
            GIP_d_te = self.compute_GIP_test(train_Y, test_Yd_impute)
            GIP_t_te = self._GIP_t
        elif self._cvs == 3:
            GIP_d_te = self._GIP_d
            test_Yt =np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
            test_Yt_impute = self._impute_Y_test(train_Y.T, test_Yt.T, St_te)
            GIP_t_te = self.compute_GIP_test(train_Y.T, test_Yt_impute)
        elif self._cvs == 4:
            test_Yd = np.zeros((self._n_drugs_te,self._n_targets))
            test_Yd_impute = self._impute_Y_test(train_Y, test_Yd, Sd_te)
            GIP_d_te = self.compute_GIP_test(train_Y, test_Yd_impute)
            test_Yt = np.zeros((self._n_drugs,self._n_targets_te))
            test_Yt_impute = self._impute_Y_test(train_Y.T, test_Yt.T, St_te)
            GIP_t_te = self.compute_GIP_test(train_Y.T, test_Yt_impute)
        return GIP_d_te, GIP_t_te
    #----------------------------------------------------------------------------------------        
    
    
    def _impute_Y_test(self, train_Y, test_Y, S_test, k=5):
        Y = np.zeros(test_Y.shape)
        # S = S_test - np.diag(np.diag(S_test))
        S = S_test
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros((train_Y.shape[0],train_Y.shape[0])))
        knns = neigh.kneighbors(1 - S, return_distance=False)
        
        for d in range(Y.shape[0]):
            ii = knns[d]
            sd = S[d,ii]
            Y[d,:] = sd@train_Y[ii,:]
            z = np.sum(sd)
            if z>0:
                Y[d,:]/=z                
        return Y
    #----------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------- 

     
    
    
    




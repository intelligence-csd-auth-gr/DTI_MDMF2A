import numpy as np
from base.mvtransbase import MultiViewTransductiveModelBase


class MDMF2A_TRANS(MultiViewTransductiveModelBase):
    """ensemble of MFAP and MFAUC """
    def __init__(self, mfap, mfauc, w):
        self.mfap = mfap
        self.mfauc = mfauc
        self.w = w        
      
        self.copyable_attrs = ['mfap','mfauc','w']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMats, targetMats, test_indices, cvs=2): 
        self._check_fit_data(intMat, drugMats, targetMats, test_indices, cvs)
        
        scores_ap = self.mfap.fit(intMat, drugMats, targetMats, test_indices, cvs)
        scores_auc = self.mfauc.fit(intMat, drugMats, targetMats, test_indices, cvs)
        scores_auc = 1/(1+np.exp(-1*scores_auc))
        scores = self.w*scores_ap + (1-self.w)*scores_auc
        return scores
    #----------------------------------------------------------------------------------------
    
    def _get_prediction_trainset(self):
        scores_ap = self.mfap._get_prediction_trainset()
        scores_auc = self.mfauc._get_prediction_trainset()
        scores_auc = 1/(1+np.exp(-1*scores_auc))
        scores = self.w*scores_ap + (1-self.w)*scores_auc
        return scores
    #----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
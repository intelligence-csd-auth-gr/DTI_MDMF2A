import numpy as np
from base.mvinbase import MultiViewInductiveModelBase


class MDMF2A(MultiViewInductiveModelBase):
    """ensemble of MFAP and MFAUC """
    def __init__(self, mfap, mfauc, w):
        self.mfap = mfap
        self.mfauc = mfauc
        self.w = w        
      
        self.copyable_attrs = ['mfap','mfauc','w']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMats, targetMats, cvs=2): 
        self._check_fit_data(intMat, drugMats, targetMats, cvs)
        
        
        self.mfap.fit(intMat, drugMats, targetMats, cvs)
        self.mfauc.fit(intMat, drugMats, targetMats, cvs)
    #----------------------------------------------------------------------------------------
    
    def predict(self, drugMatTes, targetMatTes):
        self._check_test_data(drugMatTes, targetMatTes)   
        
        scores_ap = self.mfap.predict(drugMatTes, targetMatTes)
        scores_auc = self.mfauc.predict(drugMatTes, targetMatTes)
        scores_auc = 1/(1+np.exp(-1*scores_auc))
        scores = self.w*scores_ap + (1-self.w)*scores_auc
        return scores
    #----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
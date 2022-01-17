""" Implementation of Similarity Network Fusion (SNF) based similarity combination
    [1] Zhang, Yong, Xiaohua Hu, and Xingpeng Jiang. "Multi-view clustering of microbiome samples by robust similarity network fusion and spectral clustering." IEEE/ACM transactions on computational biology and bioinformatics 14.2 (2015): 264-271.
    [2] Olayan, Rawan S., Haitham Ashoor, and Vladimir B. Bajic. "DDR: efficient computational method to predict drug–target interactions using graph mining and machine learning approaches." Bioinformatics 34.7 (2018): 1164-1173.
"""
import numpy as np


from mv.com_sim.combine_sims import Combine_Sims_Ave
from mv.mv_model_trans.SNF import SNF


class Combine_Sims_SNF(Combine_Sims_Ave):
    """ !!! SNF is used for transductive model only"""
    def __init__(self, k=3, num_iters=2, alpha = 1):
           super().__init__()
           self.k = k #the number of neareast neighbours
           self.num_iters = num_iters # the number of iterations
           self.alpha = alpha # paramter for normalized function
           
           self.copyable_attrs=self.copyable_attrs+['k','num_iters','alpha']
    #----------------------------------------------------------------------------------------     
    
    def combine(self, Ss, Y):
        if Ss.shape[0] == 1:
            S = Ss[0]
        else:
            S = SNF(Ss, self.k, self.num_iters, self.alpha)
        w = np.zeros(Ss.shape[0])
        return S, w
    #----------------------------------------------------------------------------------------
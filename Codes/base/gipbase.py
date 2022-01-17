import numpy as np

from sklearn.metrics.pairwise import rbf_kernel

""" The base class containing the methods to compute GIP similariity/Kernel"""
class GIPSimBase():
    def compute_GIP(self, Y, gamma=1.0):
        bw = gamma/(np.sum(Y)/Y.shape[0])
        gip = rbf_kernel(Y,gamma=bw)
        return gip
    #----------------------------------------------------------------------------------------
    
    def compute_GIP_test(self, Y_tr, Y_te, gamma=1.0):
        bw = gamma/(np.sum(Y_tr)/Y_tr.shape[0])
        gip = rbf_kernel(Y_te, Y_tr, gamma=bw)
        return gip
    #----------------------------------------------------------------------------------------
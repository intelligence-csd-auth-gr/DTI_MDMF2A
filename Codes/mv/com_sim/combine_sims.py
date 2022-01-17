import numpy as np
from base.csbase import Combine_Sims_Base
from sklearn.neighbors import NearestNeighbors

# class Combine_Sims_base:
#     def __init__(self):
#         pass
# #---------------------------------------------------------------------------------------- 
        
#     def combine(self, Ss, Y=None):
#         self._num_sims = Ss.shape[0]
#         w = self._compute_weights()
#         S = np.average(Ss,axis=0,weights=w) # aggragate weights based on w
#         return S, w
    
#     def _compute_weights(self):
#         w = np.full(self._num_sims,1.0/self._num_sims, dtype=float) # the sum of weights could not be one.
#         return w

class Combine_Sims_Ave(Combine_Sims_Base):
    """ 
    using equal wieghts and the weight of each similairty is 1/num_sims
    A multiple kernel learning algorithm for drug-target interaction prediction  BMC Bioinf 16
    It aslo the base class of all Combine_Sims classes
    """
    def __init__(self):
        self.copyable_attrs  = []
#---------------------------------------------------------------------------------------- 
        
    def combine(self, Ss, Y):
        self._num_sims = Ss.shape[0]
        w = self._compute_weights(Ss, Y)
        S = np.average(Ss,axis=0,weights=w) # aggragate similarity based on w
        return S, w
#---------------------------------------------------------------------------------------- 
        
    def _compute_weights(self, Ss, Y): # Ss and Y are not used in this function
        w = np.full(self._num_sims,1.0/self._num_sims, dtype=float) # the sum of weights could not be one.
        return w
#----------------------------------------------------------------------------------------   
#----------------------------------------------------------------------------------------          


class Combine_Sims_KA(Combine_Sims_Ave):
    """ 
    kernel alignment (KA) heuristics weighting method. 
    The weights of a similalrity is propotional to its aligment to the idal kernel
    A multiple kernel learning algorithm for drug-target interaction prediction  BMC Bioinf 16
    """
    def __init__(self):
        super().__init__()
#---------------------------------------------------------------------------------------- 
        
    def _compute_weights(self, Ss, Y):
        n = Ss.shape[1] # the number of rows in Ss[0]
        w = np.zeros(self._num_sims, dtype=float) 
        S_ideal = Y@Y.T
        for i in range(self._num_sims):
            w[i] = self._alignment(Ss[i],S_ideal)/(n*np.sqrt(self._alignment(Ss[i],Ss[i])))
        w = w/np.sum(w)
        return w
#----------------------------------------------------------------------------------------     
        
    def _alignment(self, S1, S2):
        A = S1*S2
        a = A.sum()
        return a     
#----------------------------------------------------------------------------------------   
#----------------------------------------------------------------------------------------   

class Combine_Sims_KA2(Combine_Sims_KA):
    """ 
    Difference with Combine_Sims_KA: S only contain k largest values in each row excluding the diagonal elements 
    """
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.copyable_attrs.append('k')
#---------------------------------------------------------------------------------------- 
        
    def _compute_weights(self, Ss, Y):
        n = Ss.shape[1] # the number of rows in Ss[0]
        w = np.zeros(self._num_sims, dtype=float) 
        neigh = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
        neigh.fit(np.zeros(Ss[0].shape))
                
        S_ideal = Y@Y.T
          
        for i in range(self._num_sims):
            S = Ss[i] - np.diag(np.diag(Ss[i])) # set diagnol elements to zeros
            knn = neigh.kneighbors(1-S, return_distance=False)
            S1 = np.zeros(S.shape)
            for j in range(S.shape[0]):
                jj = knn[j]
                S1[j, jj] = S[j, jj]
            w[i] = self._alignment(S1,S_ideal)/(n*np.sqrt(self._alignment(S1,S1)))
        w = w/np.sum(w)
        return w
#----------------------------------------------------------------------------------------     
        
    def _alignment(self, S1, S2):
        A = S1*S2
        a = A.sum()
        return a     
#----------------------------------------------------------------------------------------   
#----------------------------------------------------------------------------------------   
    
        
class Combine_Sims_Limb(Combine_Sims_Ave):
    """ 
    the weight is proprotional to the 1-local imbalance
    """
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.copyable_attrs.append('k')    
#---------------------------------------------------------------------------------------- 
        
    def _compute_weights(self, Ss, Y):
        w = np.zeros(self._num_sims, dtype=float) 
        for i in range(self._num_sims):
            S1 = Ss[i] - np.diag(np.diag(Ss[i])) # set diagnol elements to zeros
            w[i] = 1-self._cal_limb(S1, Y, self.k)
        w = w/np.sum(w)
        return w 
#----------------------------------------------------------------------------------------         
        
    def _cal_limb(self, S, Y, k):
        """ S is similarity matrix whose dignoal elememets are zeros"""
        
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros(S.shape))
        knns = neigh.kneighbors(1 - S, return_distance=False)
        
        C = np.zeros(Y.shape, dtype=float)
        for i in range(Y.shape[0]):
            ii = knns[i]
            for j in range(Y.shape[1]):
                if Y[i,j] == 1: # only consider "1" 
                    C[i,j] = k-np.sum(Y[ii,j])
        C = C/k
        milb = np.sum(C)/np.sum(Y)
        
        return milb
#----------------------------------------------------------------------------------------   
#----------------------------------------------------------------------------------------
        
    
class Combine_Sims_Limb2(Combine_Sims_Limb):
    """ 
    Difference with Combine_Sims_Limb
    the weight is proprotional to the 1/(local imbalance)
    """
    def __init__(self, k=5):
        super().__init__(k)
#---------------------------------------------------------------------------------------- 
        
    def _compute_weights(self, Ss, Y):
        w = np.zeros(self._num_sims, dtype=float) 
        for i in range(self._num_sims):
            S1 = Ss[i] - np.diag(np.diag(Ss[i])) # set diagnol elements to zeros
            w[i] = 1/self._cal_limb(S1, Y, self.k)
        w = w/np.sum(w)
        return w 
#----------------------------------------------------------------------------------------   
#----------------------------------------------------------------------------------------


class Combine_Sims_Limb3(Combine_Sims_Limb):
    """ 
    Difference with Combine_Sims_Limb
    considering the influence of similarities
    !!! This is used in the paper
    """
    def __init__(self, k=5):
        super().__init__(k)
#---------------------------------------------------------------------------------------- 
    
    def _cal_limb(self, S, Y, k):
        """ S is similarity matrix whose dignoal elememets are zeros"""
        
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros(S.shape))
        idx = np.where(S>1.0)
        knns = neigh.kneighbors(1 - S, return_distance=False)
                
        C = np.zeros(Y.shape, dtype=float)
        for i in range(Y.shape[0]):
            ii = knns[i]
            s = S[i,ii]
            z = np.sum(s)
            if z == 0:
                z=1
            C[i] = 1-s@Y[ii,:]/z
        C *= Y
        milb = np.sum(C)/np.sum(Y)
        
        return milb
#---------------------------------------------------------------------------------------- 
#----------------------------------------------------------------------------------------
        

class Combine_Sims_LowestLimb(Combine_Sims_Limb):
    """ 
    choose the one similarity with lowerest local imbalance
    """
    def __init__(self, k=5):
        super().__init__(k)
#---------------------------------------------------------------------------------------- 
    
    def _compute_weights(self, Ss, Y):
        w = np.zeros(self._num_sims, dtype=float) 
        limb_ds = np.ones(self._num_sims, dtype=float)
        for i in range(self._num_sims):
            S1 = Ss[i] - np.diag(np.diag(Ss[i])) # set diagnol elements to zeros
            limb_ds[i] = self._cal_limb(S1, Y, self.k)
        min_idx = np.argmin(limb_ds)
        w[min_idx] = 1.0
        # w = w/np.sum(w)
        return w 
#----------------------------------------------------------------------------------------

class Combine_Sims_Single(Combine_Sims_Ave):
    """ 
    Only use one single similarity
    """
    def __init__(self, ind=0):
        super().__init__()
        self.ind = ind # ind similarity is used, others' weights are set as 0
        self.copyable_attrs.append('ind')
#---------------------------------------------------------------------------------------- 
    
    def _compute_weights(self, Ss, Y):
        w = np.zeros(self._num_sims, dtype=float) 
        w[self.ind] = 1.0
        return w 
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
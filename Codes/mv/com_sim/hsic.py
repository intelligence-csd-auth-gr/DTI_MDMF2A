import numpy as np
from scipy.optimize import minimize,Bounds
from mv.com_sim.combine_sims import Combine_Sims_Ave


class HSIC(Combine_Sims_Ave):
    """ 
    Impelementation of Hilbert–Schmidt independence criterion-based multiple similarities fusion based on
    matlab codes of [1] which is availble at https://figshare.com/s/f664b36119c60e7f6f30
    [1] Ding, Yijie, Jijun Tang, and Fei Guo. "Identification of drug–target interactions via 
    dual laplacian regularized least squares with multiple kernel fusion." Knowledge-Based Systems 204 (2020): 106254.
    """
    def __init__(self, v1=2**-1, v2=2**-4, seed=0):
        super().__init__()
        self.v1 = v1
        self.v2 = v2
        self.seed = seed
        self.copyable_attrs=self.copyable_attrs+['v1','v2','seed']
        """
        v1 = [2**0, 2**-1, ..., 2**-5]
        v2 = [2**0, 2**-1, ..., 2**-5]
        """
    #----------------------------------------------------------------------------------------     
        
    def _compute_weights(self, Ss, Y):
        n = Ss.shape[1] # the number of rows in Ss[0]
        Ss1 = np.zeros(Ss.shape)
        for i in range(self._num_sims):
            Ss1[i] = self._process_sim(Ss[i])
        S_ideal = Y@Y.T # U in paper
        S_ideal = self._normalize_sim(S_ideal)
        
        H = np.eye(n)-np.ones(Ss1[0].shape, dtype=float)/n
        M = np.zeros((self._num_sims,self._num_sims), dtype=float) # the similarity between input similarity matrices
        for i in range(self._num_sims):
            for j in range(i,self._num_sims):
                mm = self._alignment(Ss1[i],Ss1[j])
                m1 = self._alignment(Ss1[i],Ss1[i])
                m2 = self._alignment(Ss1[j],Ss1[j])
                ss = mm/(np.sqrt(m1)*np.sqrt(m2))
                M[i,j] = M[j,i] = ss
        d1 = np.sum(M, axis=1)
        D1 = np.diag(d1)
        LapM = D1-M
        
        a = np.zeros(self._num_sims)
        for i in range(self._num_sims):
            kk = H@Ss1[i]@H
            aa = np.trace(kk.T@S_ideal)
            a[i] = (n**-2)*aa # n-1 in matlab code
        
        prng = np.random.RandomState(self.seed)
        w = prng.rand(self._num_sims)
        w = w/np.sum(w)
        bnds = Bounds(np.zeros(self._num_sims),np.ones(self._num_sims)) # eq.14c
        cons = ({'type': 'eq', "fun": self._constraint_eq }) # eq.14d
        res = minimize(self._f_obj, w, args=(a,LapM), method='SLSQP', bounds=bnds, constraints=cons)
        w = res.x
        return w
    #----------------------------------------------------------------------------------------  
    def _process_sim(self, S):
        # make similarity matrix symmetric
        S1 = (S+S.T)/2 
        # make similarity matrix PSD
        eig_values = np.linalg.eigvals(S)
        eig_values = np.real_if_close(eig_values) # keep the real part of eig_values
        ev_min = np.min(eig_values)
        e = max(0.0, -1.0*ev_min+1e-4)
        e1 = e.real
        S1 = S1 + e1*np.eye(S.shape[0])
        
        S1 = self._normalize_sim(S1)
        return S1
    
    def _normalize_sim(self, S):
        min_nz = np.min(S[np.nonzero(S)]) # the mininal none zero value
        S[S==0] = min_nz
        D = np.diag(S)
        D = np.sqrt(D)
        S1 = S/(np.outer(D,D)) 
        return S1
    #----------------------------------------------------------------------------------------  
    
    def _alignment(self, S1, S2):
        # same with np.trace(S1.T@S2)
        A = S1*S2
        a = A.sum()
        return a     
    #----------------------------------------------------------------------------------------  
    
    def _f_obj(self, w, a, LapM):
        #  eq.14a.
        J = -1*w@a + self.v1*w.T@LapM@w +self.v2*np.linalg.norm(w,2)**2 # last term is equalient to w@w
        return J
    #----------------------------------------------------------------------------------------
    
    def _constraint_eq(self, w):
        """
        return value must come back as 0 to be accepted 
        if return value is anything other than 0 it's rejectedas not a valid answer.
        """
        s = np.sum(w)-1
        return s
    #----------------------------------------------------------------------------------------
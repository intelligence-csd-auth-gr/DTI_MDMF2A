import numpy as np
from collections import defaultdict 

def cv_split_index_induc(intMat, cvs, num, seeds): # num: the number of folds
    """ 
    Inductive Learning, CV for Setting 4 according to 'Toward more realistic drug-target interaction predictions' 2014
    num=3: rowFold=[0,1,2] columnFold=[0,1,2], 9 test set: [r0-c0, r0-c1, r0-c2,...,r2-c2]
    """
    if cvs==1:
        print ("Inductive Learning cannot handle Setting 1 !!!")
        return None
    
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cvs == 2:  # Setting 2  
            index_d = prng.permutation(num_drugs)
            cv_index_d = np.array_split(index_d,num)   
            for i in range(num):
                train_d=np.array([], dtype=np.intc)
                for j in range(num):
                    if j!= i:
                        train_d=np.concatenate((train_d, cv_index_d[j]))
                    else:
                        test_d=cv_index_d[j]
                train_d=np.sort(train_d)
                test_d=np.sort(test_d)
                cv_data[seed].append((train_d,None,test_d,None))
                # (drug indices in training set, taraget indices in training set, drug indices in training set, target indices in test set) 
                # None: all drugs (rows) / targets (columns) are used in training/test set
        elif cvs == 3: # Setting 3 
            index_t = prng.permutation(num_targets)
            cv_index_t = np.array_split(index_t,num)
            for i in range(num):
                train_t=np.array([], dtype=np.intc)
                for j in range(num):
                    if j!= i:
                        train_t=np.concatenate((train_t, cv_index_t[j]))
                    else:
                        test_t=cv_index_t[j]
                train_t=np.sort(train_t)
                test_t=np.sort(test_t)
                cv_data[seed].append((None,train_t,None,test_t))               
        elif cvs == 4: # Setting 4
            index_d = prng.permutation(num_drugs)
            index_t = prng.permutation(num_targets)
            cv_index_d = np.array_split(index_d,num)
            cv_index_t = np.array_split(index_t,num)
            folds=[(i_r,i_c) for i_r in range(num) for i_c in range(num)]
            for i_r, i_c in folds:
                train_d = np.array([], dtype=np.intc)
                train_t = np.array([], dtype=np.intc)
                test_d = cv_index_d[i_r]
                test_t = cv_index_t[i_c]
                for j_r in range(num):
                    if(i_r != j_r):
                        train_d=np.concatenate((train_d, cv_index_d[j_r]))
                for j_c in range(num):
                    if(i_c != j_c):
                        train_t=np.concatenate((train_t, cv_index_t[j_c]))
                train_d=np.sort(train_d)
                test_d=np.sort(test_d)
                train_t=np.sort(train_t)
                test_t=np.sort(test_t)
                cv_data[seed].append((train_d,train_t,test_d,test_t))    

    # for seed, cv_index in cv_data.items():
    #     print(seed)
    #     for train_d,train_t,test_d,test_t in cv_index:
    #         print(test_d)
            
    return cv_data
#---------------------------------------------------------------------------------------------------
    
def split_train_test_set(train_d,train_t,test_d,test_t, yMat, dMat, tMat, cvs): 
    """ Single-View version 
    split the original data into training and test data according to the splitting indices."""
    if cvs==1:
        print ("Inductive Learning cannot handle Setting 1 !!!")
        return None
    
    if cvs==2:
        train_dMat = dMat[np.ix_(train_d,train_d)]
        test_dMat = dMat[np.ix_(test_d,train_d)]
        train_tMat = tMat
        test_tMat  = tMat
        train_Y = yMat[train_d,:]
        test_Y  = yMat[test_d,:]  
    elif cvs==3:
        train_dMat=dMat
        test_dMat=dMat
        train_tMat = tMat[np.ix_(train_t,train_t)]
        test_tMat  = tMat[np.ix_(test_t,train_t)] 
        train_Y = yMat[:,train_t]
        test_Y  = yMat[:,test_t] 
    elif cvs==4:
        train_dMat = dMat[np.ix_(train_d,train_d)]
        test_dMat = dMat[np.ix_(test_d,train_d)]
        train_tMat = tMat[np.ix_(train_t,train_t)]
        test_tMat  = tMat[np.ix_(test_t,train_t)] 
        train_Y = yMat[np.ix_(train_d,train_t)]
        test_Y  = yMat[np.ix_(test_d,test_t)]
        
    return train_dMat, train_tMat, train_Y, test_dMat, test_tMat, test_Y
#---------------------------------------------------------------------------------------------------


    
def split_train_test_set_mv(train_d,train_t,test_d,test_t, yMat, dMats, tMats, cvs): 
    """
    Multi-view version
    split the original data into training and test data according to the splitting indices."""
    if cvs==1:
        print ("Inductive Learning cannot handle Setting 1 !!!")
        return None
    n_dsims = dMats.shape[0]
    idx_d0 = np.arange(n_dsims, dtype = int)
    n_tsims = tMats.shape[0]
    idx_t0 = np.arange(n_tsims, dtype = int)
    if cvs==2:
        train_dMats = dMats[np.ix_(idx_d0,train_d,train_d)]
        test_dMats = dMats[np.ix_(idx_d0,test_d,train_d)]
        train_tMats = tMats
        test_tMats  = tMats
        train_Y = yMat[train_d,:]
        test_Y  = yMat[test_d,:]  
    elif cvs==3:
        train_dMats=dMats
        test_dMats=dMats
        train_tMats = tMats[np.ix_(idx_t0,train_t,train_t)]
        test_tMats  = tMats[np.ix_(idx_t0,test_t,train_t)] 
        train_Y = yMat[:,train_t]
        test_Y  = yMat[:,test_t] 
    elif cvs==4:
        train_dMats = dMats[np.ix_(idx_d0,train_d,train_d)]
        test_dMats = dMats[np.ix_(idx_d0,test_d,train_d)]
        train_tMats = tMats[np.ix_(idx_t0,train_t,train_t)]
        test_tMats  = tMats[np.ix_(idx_t0,test_t,train_t)] 
        train_Y = yMat[np.ix_(train_d,train_t)]
        test_Y  = yMat[np.ix_(test_d,test_t)]
        
    return train_dMats, train_tMats, train_Y, test_dMats, test_tMats, test_Y
#---------------------------------------------------------------------------------------------------



def cv_split_index_trans(intMat, cvs, num, seeds): # num: the number of folds
    """ 
    split cv index in transductive setting
    """
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cvs==1:
            index_p = prng.permutation(num_drugs*num_targets)
            cv_index_p = np.array_split(index_p,num)
            for i in range(num):
                train_p =np.array([], dtype=np.intc)
                for j in range(num):
                    if j!= i:
                        train_p=np.concatenate((train_p, cv_index_p[j]))
                    else:
                        test_p=cv_index_p[j]
                train_p=np.sort(train_p)
                train_p = np.unravel_index(train_p, intMat.shape) # train_p = (array([0,0,0,1,1,1]), array([0,1,2,0,1,12))
                test_p=np.sort(test_p)
                test_p = np.unravel_index(test_p, intMat.shape)
                cv_data[seed].append((train_p,test_p))
        elif cvs == 2:  # Setting 2
            index_d = prng.permutation(num_drugs)
            cv_index_d = np.array_split(index_d,num)   
            for i in range(num):
                train_d=np.array([], dtype=np.intc)
                for j in range(num):
                    if j!= i:
                        train_d=np.concatenate((train_d, cv_index_d[j]))
                    else:
                        test_d=cv_index_d[j]
                train_d=np.sort(train_d)
                test_d=np.sort(test_d)
                cv_data[seed].append((train_d,None,test_d,None))    
                # (drug indices in training set, taraget indices in training set, drug indices in training set, target indices in test set) 
                # None: all drugs (rows) / targets (columns) are used in training/test set
        elif cvs == 3: # Setting 3 
            index_t = prng.permutation(num_targets)
            cv_index_t = np.array_split(index_t,num)
            for i in range(num):
                train_t=np.array([], dtype=np.intc)
                for j in range(num):
                    if j!= i:
                        train_t=np.concatenate((train_t, cv_index_t[j]))
                    else:
                        test_t=cv_index_t[j]
                train_t=np.sort(train_t)
                test_t=np.sort(test_t)
                cv_data[seed].append((None,train_t,None,test_t))               
        elif cvs == 4: # Setting 4
            index_d = prng.permutation(num_drugs)
            index_t = prng.permutation(num_targets)
            cv_index_d = np.array_split(index_d,num)
            cv_index_t = np.array_split(index_t,num)
            folds=[(i_r,i_c) for i_r in range(num) for i_c in range(num)]
            for i_r, i_c in folds:
                train_d = np.array([], dtype=np.intc)
                train_t = np.array([], dtype=np.intc)
                test_d = cv_index_d[i_r]
                test_t = cv_index_t[i_c]
                for j_r in range(num):
                    if(i_r != j_r):
                        train_d=np.concatenate((train_d, cv_index_d[j_r]))
                for j_c in range(num):
                    if(i_c != j_c):
                        train_t=np.concatenate((train_t, cv_index_t[j_c]))
                train_d=np.sort(train_d)
                test_d=np.sort(test_d)
                train_t=np.sort(train_t)
                test_t=np.sort(test_t)
                cv_data[seed].append((train_d,train_t,test_d,test_t))    

    # for seed, cv_index in cv_data.items():
    #     print(seed)
    #     for train_d,train_t,test_d,test_t in cv_index:
    #         print(test_d)
            
    return cv_data
#---------------------------------------------------------------------------------------------------

def cv_split_index_trans_0interactions(intMat, cvs, num, seeds): # num: the number of folds
    """ 
    split cv index in transductive setting, only split 0 interactions and all 1 interactions are in training set
    this splitation only used for cvs=1
    """
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cvs==1:
            y = intMat.flatten()
            index_p1 = np.where(y==1)[0]
            index_p0 = np.where(y==0)[0]
            index_p0 = prng.permutation(index_p0)
            cv_index_p = np.array_split(index_p0,num)
            for i in range(num):
                train_p = np.array([], dtype=np.intc)
                train_p = np.concatenate((train_p, index_p1)) # add all indices of 1 interactions
                for j in range(num):
                    if j!= i:
                        train_p=np.concatenate((train_p, cv_index_p[j]))
                    else:
                        test_p=cv_index_p[j]
                train_p=np.sort(train_p)
                train_p = np.unravel_index(train_p, intMat.shape) # train_p = (array([0,0,0,1,1,1]), array([0,1,2,0,1,12))
                test_p=np.sort(test_p)
                test_p = np.unravel_index(test_p, intMat.shape)
                cv_data[seed].append((train_p,test_p))
    return cv_data
#---------------------------------------------------------------------------------------------------



""" mask the interations of test pairs """
def mask_intMat(intMat, cv_index_item, cvs):
    Y = np.copy(intMat)
    if cvs == 1:
        train_p,test_p = cv_index_item
        Y[test_p] = 0
    elif cvs == 2:
        _,_,test_d,_ = cv_index_item
        Y[test_d,:] = 0
    elif cvs == 3:
        _,_,_,test_t = cv_index_item
        Y[:,test_t] = 0
    elif cvs == 4:
        _,_,test_d,test_t = cv_index_item
        Y[test_d,:] = 0
        Y[:,test_t] = 0
    return Y
#---------------------------------------------------------------------------------------------------

def get_test_intMat_indices(intMat, cv_index_item, cvs):
    Y_te = None
    test_indices = None
    if cvs == 1:
        train_p,test_p = cv_index_item
        Y_te = intMat[test_p]
        test_indices = test_p
    elif cvs == 2:
        _,_,test_d,_ = cv_index_item
        Y_te = intMat[test_d,:]
        test_indices = test_d
    elif cvs == 3:
        _,_,_,test_t = cv_index_item
        Y_te = intMat[:,test_t]
        test_indices = test_t
    elif cvs == 4:
        _,_,test_d,test_t = cv_index_item
        Y_te = intMat[np.ix_(test_d,test_t)]
        test_indices = test_d,test_t
    return Y_te, test_indices
#---------------------------------------------------------------------------------------------------

def get_training_Y_Sds_Sts_trans(intMat, drugMat, targetMat, cv_index_item, cvs):
    """Single-view version"""
    # n_dsims = drugMats.shape[0]
    # idx_d0 = np.arange(n_dsims, dtype = int)
    # n_tsims = targetMats.shape[0]
    # idx_t0 = np.arange(n_tsims, dtype = int)
    if cvs == 1:
        train_p,_ = cv_index_item
        Y = np.zeros(intMat.shape)        
        Y[train_p] = intMat[train_p]
        Sd = drugMat
        St = targetMat
    elif cvs == 2:
        _,_,test_d,_ = cv_index_item
        all_d = np.arange(intMat.shape[0], dtype=int)
        train_d = np.setdiff1d(all_d,test_d) # training drug indices 
        
        Y = intMat[train_d,:]
        Sd = drugMat[np.ix_(train_d,train_d)]
        St = targetMat
    elif cvs == 3:
        _,_,_,test_t = cv_index_item
        all_t = np.arange(intMat.shape[1], dtype=int)
        train_t = np.setdiff1d(all_t,test_t) # training targets indices 
        
        Y = intMat[:,train_t]
        Sd=drugMat
        St = targetMat[np.ix_(train_t,train_t)]
    elif cvs == 4:
        _,_,test_d,test_t = cv_index_item
        all_d = np.arange(intMat.shape[0], dtype=int)
        train_d = np.setdiff1d(all_d,test_d) # training drug indices 
        all_t = np.arange(intMat.shape[1], dtype=int)
        train_t = np.setdiff1d(all_t,test_t) # training targets indices 
        
        Y  = intMat[np.ix_(train_d,train_t)]
        Sd = drugMat[np.ix_(train_d,train_d)]
        St = targetMat[np.ix_(train_t,train_t)]
    
    return Y, Sd, St    
#---------------------------------------------------------------------------------------------------
    
def get_training_Y_Sds_Sts_trans_mv(intMat, drugMats, targetMats, cv_index_item, cvs):
    """ Multi-view version """
    n_dsims = drugMats.shape[0]
    idx_d0 = np.arange(n_dsims, dtype = int)
    n_tsims = targetMats.shape[0]
    idx_t0 = np.arange(n_tsims, dtype = int)
    if cvs == 1:
        train_p,_ = cv_index_item
        Y = np.zeros(intMat.shape)        
        Y[train_p] = intMat[train_p]
        Sds = drugMats
        Sts = targetMats
    elif cvs == 2:
        _,_,test_d,_ = cv_index_item
        all_d = np.arange(intMat.shape[0], dtype=int)
        train_d = np.setdiff1d(all_d,test_d) # training drug indices 
        
        Y = intMat[train_d,:]
        Sds = drugMats[np.ix_(idx_d0,train_d,train_d)]
        Sts = targetMats
    elif cvs == 3:
        _,_,_,test_t = cv_index_item
        all_t = np.arange(intMat.shape[1], dtype=int)
        train_t = np.setdiff1d(all_t,test_t) # training targets indices 
        
        Y = intMat[:,train_t]
        Sds=drugMats
        Sts = targetMats[np.ix_(idx_t0,train_t,train_t)]
    elif cvs == 4:
        _,_,test_d,test_t = cv_index_item
        all_d = np.arange(intMat.shape[0], dtype=int)
        train_d = np.setdiff1d(all_d,test_d) # training drug indices 
        all_t = np.arange(intMat.shape[1], dtype=int)
        train_t = np.setdiff1d(all_t,test_t) # training targets indices 
        
        Y  = intMat[np.ix_(train_d,train_t)]
        Sds = drugMats[np.ix_(idx_d0,train_d,train_d)]
        Sts = targetMats[np.ix_(idx_t0,train_t,train_t)]
    
    return Y, Sds, Sts
#---------------------------------------------------------------------------------------------------
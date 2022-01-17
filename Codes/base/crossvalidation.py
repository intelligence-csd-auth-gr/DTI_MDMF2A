# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:34:58 2020

@author: lb
"""
import numpy as np
import time
import copy
import itertools


from base.clone import clone
from base.splitdata import *
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc, average_precision_score
from joblib import Parallel, delayed


from mv.mv_model.lcs_sv import LCS_SV
from mv.mv_model_trans.lcs_sv_trans import LCS_SV_TRANS

def train_test_model_trans(model, cvs, intMat, drugMats, targetMats, cv_index_item, param=None):
    Y = mask_intMat(intMat, cv_index_item, cvs)
    test_Y, test_indices = get_test_intMat_indices(intMat, cv_index_item, cvs)
    m = clone(model)
    if param != None:
        m.set_params(**param)
    tic=time.time()
    scores = m.fit(Y, drugMats, targetMats, test_indices, cvs)
    run_time = time.time()-tic
    aupr_val, auc_val = cal_metrics(scores, test_Y)
    return aupr_val, auc_val, run_time
#---------------------------------------------------------------------------------------------------

def train_test_model_trans_scores(model, cvs, intMat, drugMats, targetMats, cv_index_item, param=None):
    """ return prediction scores instead of auc, aupr and time """
    Y = mask_intMat(intMat, cv_index_item, cvs)
    test_Y, test_indices = get_test_intMat_indices(intMat, cv_index_item, cvs)
    m = clone(model)
    if param != None:
        m.set_params(**param)
    scores = m.fit(Y, drugMats, targetMats, test_indices, cvs)
    return scores, test_indices
#---------------------------------------------------------------------------------------------------
    
def train_test_model_induc(model, cvs, intMat, drugMats, targetMats, cv_index_item, param=None):
    train_d, train_t, test_d, test_t = cv_index_item
    if drugMats.ndim == 2: # single similarity 
        train_dMats, train_tMats, train_Y, test_dMats, test_tMats, test_Y = split_train_test_set(train_d,train_t,test_d,test_t, intMat, drugMats, targetMats,cvs)
    elif drugMats.ndim == 3: # multiple similarities
        train_dMats, train_tMats, train_Y, test_dMats, test_tMats, test_Y = split_train_test_set_mv(train_d,train_t,test_d,test_t, intMat, drugMats, targetMats,cvs)
    m = clone(model)
    if param != None:
        m.set_params(**param)
    tic=time.time()
    m.fit(train_Y, train_dMats, train_tMats, cvs)
    scores = m.predict(test_dMats, test_tMats) 
    run_time = time.time()-tic
    aupr_val, auc_val = cal_metrics(scores, test_Y)
    return aupr_val, auc_val, run_time 
#---------------------------------------------------------------------------------------------------

def cross_validate_parallel(model, cvs, num, intMat, drugMats, targetMats, seeds=[0], out_file_name=None, n_jobs=5, isInductive=True):
    """    
    parammeters
    ----------
    model : multi view model. 
    cvs : int, cvs = 2,3 or 4
        The setting of cross validation.
    num : int 
        the number of the fold for CV.
    intMat : ndarray
        interaction matrix.
    drugMats : ndarray
        drug similairty matrices.
    targetMats : ndarray
        target similairty matrices.
    seed: int
        the seed used to split the dataset. The default is None.
    out_file_name : string, optional
        the out put file out detail results of CV. If it is None, do not out put detial results. The default is None.
        
    Returns
    -------
    auprs : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUPR results for each fold.
    aucs : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUC results for each fold.
    times : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUC results for each fold.
    """
    aupr, auc, times = [], [], []
    if isInductive:
        cv_data = cv_split_index_induc(intMat, cvs, num, seeds)
    else: # Transductive
        cv_data = cv_split_index_trans(intMat, cvs, num, seeds)
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tParams\n")
    cv_index = []
    for seed, cv_index1 in cv_data.items():
        cv_index.extend(cv_index1)  
        
    if isInductive:    
        result_list = Parallel(n_jobs=n_jobs)(delayed(train_test_model_induc)(model, cvs, intMat, drugMats, targetMats, cv_index_item) for cv_index_item in cv_index)
    else: # Transductive
        result_list = Parallel(n_jobs=n_jobs)(delayed(train_test_model_trans)(model, cvs, intMat, drugMats, targetMats, cv_index_item) for cv_index_item in cv_index)

    for aupr_val, auc_val, run_time in result_list:
        aupr.append(aupr_val)
        auc.append(auc_val)
        times.append(run_time)
        if out_file_name is not None:
            with open(out_file_name,"a") as f:
                f.write("{:.6f}\t{:.6f}\t{:.6f}\n".format(aupr_val, auc_val, run_time)) 
    auprs = np.array(aupr, dtype=np.float64)
    aucs = np.array(auc, dtype=np.float64)
    times = np.array(times, dtype=np.float64)
    return auprs, aucs, times
#---------------------------------------------------------------------------------------------------    

def cross_validate(model, cvs, num, intMat, drugMats, targetMats, seeds=[0], out_file_name=None, isInductive=True):
    aupr, auc, times = [], [], []
    if isInductive:
        cv_data = cv_split_index_induc(intMat, cvs, num, seeds)
    else: # Transductive
        cv_data = cv_split_index_trans(intMat, cvs, num, seeds)
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tParams\n")
            
    for seed, cv_index in cv_data.items():
        for cv_index_item in cv_index:
            if isInductive:
                aupr_val, auc_val, run_time = train_test_model_induc(model, cvs, intMat, drugMats, targetMats, cv_index_item)
            else: # Transductive
                aupr_val, auc_val, run_time = train_test_model_trans(model, cvs, intMat, drugMats, targetMats, cv_index_item)
            aupr.append(aupr_val)
            auc.append(auc_val)
            times.append(run_time)
            if out_file_name is not None:
                with open(out_file_name,"a") as f:
                    f.write("{:.6f}\t{:.6f}\t{:.6f}\n".format(aupr_val, auc_val, run_time)) 
                
    auprs = np.array(aupr, dtype=np.float64)
    aucs = np.array(auc, dtype=np.float64)
    times = np.array(times, dtype=np.float64)
    return auprs, aucs, times
#---------------------------------------------------------------------------------------------------    

def cross_validate_various_params_parallel(model, cvs, num, intMat, drugMats, targetMats, seeds=[0], out_file_name=None, n_jobs=5, isInductive=True, param_list=None, metric='aupr', cs_param_flag=False):
    best_aupr, best_auc, best_run_time = 0, 0, 0
    best_param = dict()
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tParams\n")     
    num_params = len(param_list)
    if num_params < n_jobs:
        n_jobs = num_params
    result_list = Parallel(n_jobs=n_jobs)(delayed(cross_validate_with_param)(model, cvs, num, intMat, drugMats, targetMats, seeds, None, isInductive, param, cs_param_flag) for param in param_list)
    
    for i in range(len(result_list)):
        aupr, auc, run_time = result_list[i]
        param = param_list[i]
        if out_file_name is not None:                
            with open(out_file_name,"a") as f:
                f.write('{:.6f}\t{:.6f}\t{:6f}\t{}\n'.format(aupr, auc, run_time, param)) 
        flag = False # if change the best..
        if metric == 'aupr' and aupr > best_aupr:
            flag = True
        if metric == 'auc' and auc > best_auc:
            flag = True 
        if metric == 'auc_aupr' and auc+aupr > best_auc+best_aupr:
            flag = True
        if flag:
            best_aupr = aupr
            best_auc = auc
            best_run_time = run_time
            best_param = param   
    if out_file_name is not None:                
        with open(out_file_name,"a") as f:
            f.write('\n\nOptimal parammeter setting:\n')
            f.write('{:.6f}\t{:.6f}\t{:6f}\t{}\n'.format(best_aupr,best_auc, best_run_time, best_param))      
    return best_aupr, best_auc, best_run_time, best_param, result_list
#---------------------------------------------------------------------------------------------------
    

def cross_validate_with_param(model, cvs, num, intMat, drugMats, targetMats, seeds=[0], out_file_name=None, isInductive=True, param=None, cs_param_flag=False):
    if isinstance(model, LCS_SV) or isinstance(model, LCS_SV_TRANS):
        if cs_param_flag == True:
            model.cs_model.set_params(**param)
        else:
            model.sv_model.set_params(**param)
    else:
        model.set_params(**param)
    auprs, aucs, run_times = cross_validate(model, cvs, num, intMat, drugMats, targetMats, seeds, None, isInductive)
    return auprs.mean(), aucs.mean(), run_times.mean()
#---------------------------------------------------------------------------------------------------    

def cross_validate_CV_0_socres(model, cvs, num, intMat, drugMats, targetMats, seeds=[0], out_file_name=None, isInductive=True):
    """ cross validate for find new DTIs.
        This funciton only for S1 predction setting (cvs=1) and transductive learning setting
        In each CV interation, use 9 folds 0 interactions and all 1 interactions as training set and the remianing 1 fold 0 interactions as test set.
        
        This new DTI discovery procedure is used by following papers:
           [1] DDR: efficient computational method to predict drug–target interactions using graph mining and machine learning approaches
           [2] DTiGEMS+: drug–target interaction prediction using graph embedding, graph mining, and similarity-based techniques
    """
    if cvs != 1:
        print('cvs!=1 in cross_validate_CV_0_socres !!')
        return
    if isInductive:
        pass
    else: # Transductive
        cv_data = cv_split_index_trans_0interactions(intMat, cvs, num, seeds)
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tParams\n")
    
    S = np.ones(intMat.shape)*-100 # prediction matrix, default values (prediction values for 1 interactions) are -100
    for seed, cv_index in cv_data.items():
        for cv_index_item in cv_index:
            if isInductive:
                pass
            else: # Transductive
                scores, test_indices = train_test_model_trans_scores(model, cvs, intMat, drugMats, targetMats, cv_index_item)
            S[test_indices] = scores
                
    return S
#---------------------------------------------------------------------------------------------------  

def cross_validate_CV_0_socres_parallel(model, cvs, num, intMat, drugMats, targetMats, seeds=[0], out_file_name=None, n_jobs=5, isInductive=True):
    """ cross validate for find new DTIs.
        This funciton only for S1 predction setting (cvs=1) and transductive learning setting
        In each CV interation, use 9 folds 0 interactions and all 1 interactions as training set and the remianing 1 fold 0 interactions as test set.
        
        This new DTI discovery procedure is used by following papers:
           [1] DDR: efficient computational method to predict drug–target interactions using graph mining and machine learning approaches
           [2] DTiGEMS+: drug–target interaction prediction using graph embedding, graph mining, and similarity-based techniques
    """
    if cvs != 1:
        print('cvs!=1 in cross_validate_CV_0_socres !!')
        return
    if isInductive:
        pass
    else: # Transductive
        cv_data = cv_split_index_trans_0interactions(intMat, cvs, num, seeds)
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tParams\n")
    cv_index = []
    for seed, cv_index1 in cv_data.items():
        cv_index.extend(cv_index1)
    
    S = np.ones(intMat.shape)*-100 # prediction matrix, default values (prediction values for 1 interactions) are -100
    if isInductive:
        pass
    else: # Transductive
        result_list = Parallel(n_jobs=n_jobs)(delayed(train_test_model_trans_scores)(model, cvs, intMat, drugMats, targetMats, cv_index_item) for cv_index_item in cv_index)

    for scores, test_indices in result_list:
        S[test_indices] = scores
                
    return S
#---------------------------------------------------------------------------------------------------  

def cross_validate_CV_0_socres_various_params_parallel(model, cvs, num, intMat, drugMats, targetMats, seeds=[0], out_file_name=None, n_jobs=5, isInductive=True, param_list=None, metric='aupr', cs_param_flag=False):
    # best_aupr, best_auc, best_run_time = 0, 0, 0
    # best_param = dict()
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tParams\n")     
    num_params = len(param_list)
    if num_params < n_jobs:
        n_jobs = num_params
    scores_list = Parallel(n_jobs=n_jobs)(delayed(cross_validate_CV_0_socres_with_param)(model, cvs, num, intMat, drugMats, targetMats, seeds, None, isInductive, param, cs_param_flag) for param in param_list)   
    return scores_list
#---------------------------------------------------------------------------------------------------
    

def cross_validate_CV_0_socres_with_param(model, cvs, num, intMat, drugMats, targetMats, seeds=[0], out_file_name=None, isInductive=True, param=None, cs_param_flag=False):
    if isinstance(model, LCS_SV) or isinstance(model, LCS_SV_TRANS):
        if cs_param_flag == True:
            model.cs_model.set_params(**param)
        else:
            model.sv_model.set_params(**param)
    else:
        model.set_params(**param)
    scores = cross_validate_CV_0_socres(model, cvs, num, intMat, drugMats, targetMats, seeds, None, isInductive)
    return scores
#---------------------------------------------------------------------------------------------------    



def cross_validate_innerCV_params(model, cvs, num, intMat, drugMats, targetMats, seeds=[0], out_file_name=None, n_jobs=5, isInductive=True, param_list=None, metric='aupr'):
    inner_num = 5 # the number of fold for inner CV
    if intMat.shape[0]+intMat.shape[1]<100: # small size dataset as nr
        inner_num = 10
    if cvs == 4:
        inner_num = 2
        if intMat.shape[0]+intMat.shape[1]<100: # small size dataset as nr
            inner_num = 3
    aupr, auc, times, opti_params = [], [], [], []
    if isInductive:
        cv_data = cv_split_index_induc(intMat, cvs, num, seeds)
    else: # Transductive
        cv_data = cv_split_index_trans(intMat, cvs, num, seeds)
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tinner_cv_time\tParams\n")
    
    for seed, cv_index in cv_data.items():
        for cv_index_item in cv_index:
            if isInductive:
                # get the training data
                train_d,train_t,test_d,test_t = cv_index_item
                Sd, St, Y, _, _, _ = split_train_test_set(train_d,train_t,test_d,test_t, intMat, drugMats, targetMats,cvs)
                tic=time.time()
                _,_,_, params, _ = cross_validate_various_params_parallel(model, cvs, inner_num, Y, Sd, St, [0], None, n_jobs, isInductive, param_list,metric)
                inner_cv_time = time.time()-tic
                aupr_val, auc_val, run_time = train_test_model_induc(model, cvs, intMat, drugMats, targetMats, cv_index_item, params)
            else: # Transductive
                # get the Y, Sds, Sts only for training set
                Y, Sd, St = get_training_Y_Sds_Sts_trans(intMat, drugMats, targetMats, cv_index_item, cvs)
                tic=time.time()
                _,_,_, params, _ = cross_validate_various_params_parallel(model, cvs, inner_num, Y, Sds, Sts, [0], None, n_jobs, isInductive, param_list,metric)
                inner_cv_time = time.time()-tic
                aupr_val, auc_val, run_time = train_test_model_trans(model, cvs, intMat, drugMats, targetMats, cv_index_item, params)
            aupr.append(aupr_val)
            auc.append(auc_val)
            times.append(run_time)
            opti_params.append(params)
            if out_file_name is not None:
                with open(out_file_name,"a") as f:
                    f.write("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n".format(aupr_val, auc_val, run_time, inner_cv_time, params)) 
    auprs = np.array(aupr, dtype=np.float64)
    aucs = np.array(auc, dtype=np.float64)
    times = np.array(times, dtype=np.float64)
    
    return auprs, aucs, times, opti_params
#---------------------------------------------------------------------------------------------------  

def get_top_ranked_dtpair(scores, intMat ,drug_names, target_names, n=10, flag=2):
    """
    Parameters
    ----------
    scores : 2-d array
        predicting scores.
    intMat : 2-d array
        interaction matrix.
    target_names : list of string
        drugs name.
    names_t : list of string
        target name.
    n : int, optional
        the number of top ranked pair to return. The default is 10.
    flag : {0,1,2}, optional
        0:pairs with interaction=0 , 1:pairs with interaction=1, 2 or other values: all paris . The default is 0.

    Returns
    -------
    list of pairs with scores, interaction
    """
    s = scores.flatten()
    sort_index = s.argsort()[::-1] # the sorted index of s in desending order 
    c = 0
    pairs = []
    for index in sort_index:
        i,j = np.unravel_index(index,scores.shape)
        if flag == 0 and intMat[i,j] !=0:
            continue
        if flag == 1 and intMat[i,j] !=1:
            continue
        pairs.append((drug_names[i],target_names[j],scores[i,j],intMat[i,j]))
        c+=1
        if c==n:
            break
    return pairs
#---------------------------------------------------------------------------------------------------


def cal_aupr(scores_1d, labels_1d):
    """ scores_1d and labels_1d is 1 dimensional array """
    # prec, rec, thr = precision_recall_curve(labels_1d, scores_1d)
    # aupr_val =  auc(rec, prec)
    # This kind of calculation is not right
    aupr_val = average_precision_score(labels_1d,scores_1d)
    if np.isnan(aupr_val):
        aupr_val=0.0
    return aupr_val

        
def cal_auc(scores_1d, labels_1d):    
    """ scores_1d and labels_1d is 1 dimensional array """
    fpr, tpr, thr = roc_curve(labels_1d, scores_1d)
    auc_val = auc(fpr, tpr)
    # same with:    roc_auc_score(labels,scores)
    if np.isnan(auc_val):
        auc_val=0.5
    return auc_val


def cal_metrics(scores, labels):
    """ scores and labels are 2 dimensional matrix, return aucpr, auc"""
    scores_1d=scores.flatten()
    labels_1d=labels.flatten()
    aupr_val=cal_aupr(scores_1d, labels_1d)
    auc_val= cal_auc(scores_1d, labels_1d)
    return aupr_val, auc_val
import os
import time
import sys
import numpy as np

from base.crossvalidation import *
from base.loaddata import * 
from base.splitdata import *


from mf.nrlmf import NRLMF
from model_trans.mf.nrlmf_trans import NRLMF_TRANS


from mv.com_sim.combine_sims import *


from mv.mv_model.mdwmfap2 import MDMFAUPR
from mv.mv_model.mdwmfauc import MDMFAUC
from mv.mv_model.dwmf2a import MDMF2A

from mv.mv_model_trans.mwdmfap2_trans import MDMFAUPR_TRANS
from mv.mv_model_trans.mwdmfauc_trans import MDMFAUC_TRANS
from mv.mv_model_trans.dwmf2a_trans import MDMF2A_TRANS





def initialize_model_mv(method, cvs=2):
    a = 0.25
    if method == 'MDMFAUPR':
        model = MDMFAUPR(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), cs_model=Combine_Sims_Limb3(), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, lambda_M=0.005, n_b=21, n_wz=5, num_neg=1, max_iter=100, seed=0, is_comLoss=1, isAGD=False, is_sparseM = True)
    elif method == 'MDMFAUC':
        model = MDMFAUC(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), cs_model=Combine_Sims_Limb3(), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, lambda_M=0.005, n_wz=5, num_neg=1, max_iter=100, seed=0, sfun=2, is_comLoss=0, isAGD=True, is_sparseM = True)  
    elif method == 'MDMF2A':
        mfap = MDMFAUPR(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), cs_model=Combine_Sims_Limb3(), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, lambda_M=0.005, n_b=21, n_wz=5, num_neg=1, max_iter=100, seed=0, is_comLoss=1, isAGD=False, is_sparseM = True)
        mfauc = MDMFAUC(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), cs_model=Combine_Sims_Limb3(), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, lambda_M=0.005, n_wz=5, num_neg=1, max_iter=100, seed=0, sfun=2, is_comLoss=0, isAGD=True, is_sparseM = True) 
        model = MDMF2A(mfap, mfauc, w=0.5)
        
    elif method == 'MDMFAUPR_TRANS':
        model = MDMFAUPR_TRANS(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), cs_model=Combine_Sims_Limb3(), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, lambda_M=0.005, n_b=21, n_wz=5, num_neg=1, max_iter=100, seed=0, is_comLoss=1, isAGD=False, is_sparseM = True) # is_sparseM = False
    elif method == 'MDMFAUC_TRANS':
        model = MDMFAUC_TRANS(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), cs_model=Combine_Sims_Limb3(), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, lambda_M=0.005, n_wz=5, num_neg=1, max_iter=100, seed=0, sfun=2, is_comLoss=0, isAGD=True, is_sparseM = True) # is_sparseM = False
    elif method == 'MDMF2A_TRANS':
        mfap = MDMFAUPR_TRANS(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), cs_model=Combine_Sims_Limb3(), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, lambda_M=0.005, n_b=21, n_wz=5, num_neg=1, max_iter=100, seed=0, is_comLoss=1, isAGD=False, is_sparseM = True)
        mfauc = MDMFAUC_TRANS(K=5, K2=5, metric=0, Ts=np.arange(0.1,1.1,0.1), cs_model=Combine_Sims_Limb3(), num_factors=50, theta=0.1, lambda_d=a, lambda_t=a, lambda_r=a, lambda_M=0.005, n_wz=5, num_neg=1, max_iter=100, seed=0, sfun=2, is_comLoss=0, isAGD=True, is_sparseM = True) 
        model = MDMF2A_TRANS(mfap, mfauc, w=0.5)
    else:
        raise RuntimeError("The method name: {} has not been defined initialize_model_mv function!!".format(method))
    return model
#----------------------------------------------------------------------------------------


def initialize_model_cs(method, cvs =2):
    if method == 'Combine_Sims_Ave':
        model = Combine_Sims_Ave()
    elif method == 'Combine_Sims_KA':
        model = Combine_Sims_KA()
    elif method == 'Combine_Sims_Limb3':
        model = Combine_Sims_Limb3(k = 5)
    elif method == 'Combine_Sims_Single':
        model = Combine_Sims_Single(ind=0)
    else:
        raise RuntimeError("The method name: {} has not been defined initialize_model_cs function!!".format(method))
    return model    
#----------------------------------------------------------------------------------------


if __name__ == "__main__": 
    # !!! my_path should be change to the path of the project in your machine            
    my_path = 'F:\envs\GitHub_MF2A_MDMF2A'
    n_jobs = 20 # set the n_jobs = 20 if possible
    
    sys.path.append(os.path.join(my_path, 'In_Trans_DTI'))  
    
    data_dir =  os.path.join(my_path, 'datasets_mv') #'F:\envs\DPI_Py37\Codes_Py3\datasets' #'data'
    output_dir = os.path.join(my_path, 'output', 'Test') # num_factors50 #'F:\envs\DPI_Py37\Codes_Py3\output' 
    seeds = [0,1]
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir) 
    out_summary_file = os.path.join(output_dir, "summary_result"+ time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()) +".txt")

      
    """ run cross validation on a model with best parameters setting retrived from file """
    param_folder = 'method_params_VP2 data1 is_sparseM=True'
    cmd ="Method\tcvs\tDataSet\tAUPR\tAUC\tTime\tTotalTime"
    print(cmd)
    with open(out_summary_file,"w") as f:
        f.write(cmd+"\n")
    for method_ori in ['MDMF2A']:  
        for cvs in [1,2,3,4]: # 1,2,3,4
            num = 10
            if cvs == 4:
                num = 3
            if cvs == 1 and '_TRANS' not in method_ori:
                method = method_ori+'_TRANS'
            else:
                method = method_ori
            
            if '_TRANS' in method:
                isInductive = False
            else:
                isInductive = True   
            model = initialize_model_mv(method)  # parameters could be changed in "initialize_model" function    
            
            if 'MDMF2A' in method:
                method_ap = method.replace('MDMF2A', 'MDMFAUPR')
                vp_best_param_file_ap = os.path.join(data_dir, param_folder,method_ap+'_best_param.txt')
                dict_params_ap = get_params2(vp_best_param_file_ap, num_key=3) # read parameters from file
            
                method_auc = method.replace('MDMF2A', 'MDMFAUC')
                vp_best_param_file_auc = os.path.join(data_dir, param_folder ,method_auc+'_best_param.txt')
                dict_params_auc = get_params2(vp_best_param_file_auc, num_key=3) # read parameters from file
            vp_best_param_file = os.path.join(data_dir, param_folder, method+'_best_param.txt')
            dict_params = get_params2(vp_best_param_file, num_key=3) # read parameters from file
            
            
            for dataset in ['gpcr1']:  # options: 'nr1','gpcr1','ic1','e1', 'luo' 
                if dataset == 'luo':  seeds = [0]
                else:  seeds = [0,1]
                out_file_name= os.path.join(output_dir, 'Best_parameters_'+method+"_"+"S"+str(cvs)+"_"+dataset+".txt") 
                intMat, drugMats, targetMats, Dsim_names, Tsim_names = load_datasets(dataset, data_dir ,'low4Limb') # ,'Original','low4Limb'
                
                if 'MF2A' in method:
                    param_ap = dict_params_ap[(method_ap, str(cvs), dataset)]
                    model.mfap.set_params(**param_ap)
                    
                    param_auc = dict_params_auc[(method_auc, str(cvs), dataset)]
                    model.mfauc.set_params(**param_auc)
                    
                param = dict_params[(method, str(cvs), dataset)]
                model.set_params(**param)
                
                tic = time.time()
                # auprs, aucs, run_times = cross_validate(model, cvs, num, intMat, drugMats, targetMats, seeds, out_file_name, isInductive)
                auprs, aucs, run_times = cross_validate_parallel(model, cvs, num, intMat, drugMats, targetMats, seeds, out_file_name, n_jobs, isInductive)
                cmd = "{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}".format(method,cvs,dataset,auprs.mean(),aucs.mean(),run_times.mean(),time.time()-tic,param)
                print(cmd)
                with open(out_summary_file,"a") as f:
                    f.write(cmd+"\n")    
   
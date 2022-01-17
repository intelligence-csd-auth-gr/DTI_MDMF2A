import os
import numpy as np
import json

def load_data_from_file(dataset, folder):
    """ single similarity data """
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        next(inf) # skip the first line
        int_array = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, dataset+"_simmat_dc.txt"), "r") as inf:  # the drug similarity file
        next(inf) # skip the first line
        drug_sim = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, dataset+"_simmat_dg.txt"), "r") as inf:  # the target similarity file
        next(inf) # skip the first line
        target_sim = [line.strip("\n").split()[1:] for line in inf]

    intMat = np.array(int_array, dtype=np.float64).T    # drug-target interaction matrix where the row is drug and the column is target
    drugMat = np.array(drug_sim, dtype=np.float64)      # drug similarity matrix
    targetMat = np.array(target_sim, dtype=np.float64)  # target similarity matrix
    return intMat, drugMat, targetMat
#----------------------------------------------------------------------------------------

def built_multiple_similarity_matrix(simfile_path, sim_name_file, length):
    # length is the number of drugs or targets
    sim_files = np.loadtxt(sim_name_file, delimiter='\n',dtype=str, ndmin=1) # ndmin=1 avoid the error when only one line in the file
    ns = sim_files.size
    sims = np.zeros(shape=(ns, length, length)) # sims[i] is the i-th similarity matrix
    for i in range(ns):
        file = os.path.join(simfile_path,sim_files[i])
        sims[i] = np.loadtxt(file, delimiter='\t',dtype=np.float64,skiprows=1, usecols=range(1,length+1))
        
    return sim_files, sims
#----------------------------------------------------------------------------------------

def load_datasets(dataset, file_path, method='all'):
    """ multiple similarities data 
        method is the type of similarities used 
    """
    if dataset[-1] == '1' or dataset[-1]=='4': # the updated interaction matrix
        sim_data_name = dataset[:-1]
    else:
        sim_data_name = dataset
    data_path = os.path.join(file_path, sim_data_name)
    interaction_file = os.path.join(data_path, dataset+"_admat_dgc.txt") # interaction matrix file # "Input/"+data+"/"+data+"_admat_dgc.txt"
    drugs, targets = get_drugs_targets_names_mvdata(interaction_file)
    intMat = np.loadtxt(interaction_file, delimiter='\t',dtype=float ,skiprows=1, usecols=range(1,len(drugs)+1))
    intMat = intMat.T
    
    Dsimfile_path = os.path.join(data_path, 'Dsim')
    if method == 'all':
        Dsim_name_file = os.path.join(Dsimfile_path,'allDsim_files.txt')  # 'allDsim_files.txt'
    elif method == '' or method == 'FSS':
        Dsim_name_file = os.path.join(Dsimfile_path,'selected_Dsim_files.txt') # forward feature selection (DTiGEMS+ paper) 
    else:
        Dsim_name_file = os.path.join(Dsimfile_path,'selected_Dsim_files_'+method+'.txt')
    Dsim_names, Dsims = built_multiple_similarity_matrix(Dsimfile_path, Dsim_name_file, len(drugs))
    
    Tsimfile_path = os.path.join(data_path, 'Tsim')
    if method == 'all':
        Tsim_name_file = os.path.join(Tsimfile_path,'allTsim_files.txt')  # 'allTsim_files.txt'
    elif method == '' or method == 'FSS':
        Tsim_name_file = os.path.join(Tsimfile_path,'selected_Tsim_files.txt') # forward feature selection (DTiGEMS+ paper)
    else:
        Tsim_name_file = os.path.join(Tsimfile_path,'selected_Tsim_files_'+method+'.txt')
    Tsim_names, Tsims = built_multiple_similarity_matrix(Tsimfile_path, Tsim_name_file, len(targets))
    
    return intMat, Dsims, Tsims, Dsim_names, Tsim_names
#----------------------------------------------------------------------------------------


def get_drugs_targets_names(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        firstRow = next(inf)
        drugs = firstRow.strip("\n").split()
        targets = [line.strip("\n").split()[0] for line in inf]
    return drugs, targets
#----------------------------------------------------------------------------------------
    
def get_drugs_targets_names_mvdata(interaction_file):
    with open(interaction_file, "r") as inf:
        firstRow = next(inf)
        drugs = firstRow.strip("\n").split()
        targets = [line.strip("\n").split()[0] for line in inf]
    return drugs, targets
#----------------------------------------------------------------------------------------

            

def get_params(param_file):
    """ read params for each fold"""
    params = []
    with open(param_file, "r", encoding='utf-8') as inf:
        next(inf) # skip the 1st line
        for line in inf:
            ss = line.strip().split('\t')
            # ss[3] = ss[3].replace(", 'avg': True","" ) # remove the avg parameter for BLMNII
            s = ss[-1].replace("'", "\"")
            s = s.replace('True','true').replace('False','false')
            param = json.loads(s)
            params.append(param)
    return params
#----------------------------------------------------------------------------------------

def get_params2(param_file, num_key=3):
    """ read params for each method
        num_key: the number of elements compose the key 
    """
    params_dict = dict()
    with open(param_file, "r", encoding='utf-8') as inf:
        next(inf) # skip the 1st line
        for line in inf:
            ss = line.strip().split('\t')
            key = tuple(ss[:num_key])
            # ss[3] = ss[3].replace(", 'avg': True","" ) # remove the avg parameter for BLMNII
            s = ss[-1].replace("'", "\"")
            s = s.replace('True','true').replace('False','false')
            param = json.loads(s)
            params_dict[key] = param
    return params_dict
#----------------------------------------------------------------------------------------

def param_dict2str(param):
    """ transform the parameter dict to string """
    s = ""
    for key, value in param.items():
        s += key+str(value)+"_"
    s = s[:-1] # delete the last "_"
    return s

#----------------------------------------------------------------------------------------


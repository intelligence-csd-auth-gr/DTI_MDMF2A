import numpy as np
import copy	


"""
the paper "Multi-View Clustering of Microbiome Samples by Robust Similarity Network Fusion and Spectral Clustering" descrbes the detail of SNF algorithm
The P^v is(similar) the Wall[i] after normalization (sum of rows is 1)
The S^v is the newW[i] containing simialrities of kNNs 
"""


def FindDominantSet(W,K):
    m,n = W.shape
    DS = np.zeros((m,n))
    for i in range(m):
        index =  np.argsort(W[i,:])[-K:] # get the closest K neighbors 
        DS[i,index] = W[i,index] # keep only the nearest neighbors 

	#normalize by sum 
    B = np.sum(DS,axis=1)
    # B[B == 0] = 1 # B can not equal to zero becasue it will be used as denominator
    B = B.reshape(len(B),1)
    DS = DS/B
    return DS
#---------------------------------------------------------------------------------------------------


def normalized(W,alpha):
	m,n = W.shape
	W = W+alpha*np.identity(m)
	return (W+np.transpose(W))/2
#---------------------------------------------------------------------------------------------------



def SNF(Wall1,K,t,alpha=1):
    # Wall1 a list of similarity matrix
    # K is the number of neareast neighbours
    # t is the number of iterations of the algrithm
    # alpha is paramters for normalized function
	
    Wall = np.copy(Wall1)
    
    C = len(Wall) # C is the number of similarity matrices
    m,n = Wall[0].shape
    # normalize each similarity matrix to ensure the sum of each row is 1
    for i in range(C):
        B = np.sum(Wall[i],axis=1) # B is the sum of each row
        # B[B == 0] = 1 # B can not equal to zero becasue it will be used as denominator
        len_b = len(B)
        B = B.reshape(len_b,1)
        Wall[i] = Wall[i]/B # normalize Wall[i] to ensure the sum of each row is 1
        Wall[i] = (Wall[i]+np.transpose(Wall[i]))/2


    # newW is the sparsified Wall that only perserves k-largest similairty values (kNNs of each row instance) in each row
    newW = []
    for i in range(C):
        newW.append(FindDominantSet(Wall[i],K))
		

    # Wsum is the sum of all similarity matrices
    Wsum = np.zeros((m,n))
    for i in range(C):
        Wsum += Wall[i]

    for iteration in range(t):
        Wall0 = []
        for i in range(C):
            temp = np.dot(np.dot(newW[i], (Wsum - Wall[i])),np.transpose(newW[i]))/(C-1)
            Wall0.append(temp)
            
        for i in range(C):
            Wall[i] = normalized(Wall0[i],alpha)

        Wsum = np.zeros((m,n))
        for i in range (C):
            Wsum+=Wall[i]

    W = Wsum/C
    B = np.sum(W,axis=1)
    B = B.reshape(len(B),1)
    W/=B
    W = (W+np.transpose(W)+np.identity(m))/2
    return W
#---------------------------------------------------------------------------------------------------






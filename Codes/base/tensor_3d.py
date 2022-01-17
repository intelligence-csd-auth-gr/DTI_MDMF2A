# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:39:32 2021

@author: lb
"""
import numpy as np

""" This file contains some operations for 3rd-order tensor"""


def tensorMatricize(Xs, n=1):
    """
    compute and return the mode-n matricization of tensor Xs
    Parameters
    ----------
    Xs : TYPE 3d array
        DESCRIPTION the input tensor .
    n : TYPE integer, 
        DESCRIPTION mode of matricization. The default is 1.

    Returns 
    -------
    X_ the mode-n matricization of tensor Xs
    """
    n1, n2, n3 = Xs.shape 
    if n==1:
        X = Xs.transpose((0,2,1)).reshape(n1,n3*n2)
    elif n==2:
        X = Xs.transpose((1,2,0)).reshape(n2,n3*n1)
    elif n==3:
        X = Xs.transpose((2,1,0)).reshape(n3,n2*n1)
    return X
#----------------------------------------------------------------------------------------

def factorMatrics2Tensor(A,B,C):
    """
    Compute the [[A,B,C]] for the CP (DECOMP/PARAFAC) decomposition
    Parameters
    ----------
    A : TYPE 2d array shape=(n1,r)
        DESCRIPTION the matrix factors.
    B : TYPE 2d array shape=(n2,r)
        DESCRIPTION the matrix factors.
    C : TYPE 2d array shape=(n3,r)
        DESCRIPTION the matrix factors.

    Returns
    -------
    TYPE 3d array
        DESCRIPTION the computed tensor.
    """
    return np.einsum('il,jl,kl->ijk', A, B, C)




"""
# Test matricize based on the example of https://www.slideshare.net/panisson/tensor-decomposition-with-python   Slide No.20
Xs = np.arange(24).reshape(3,4,2)
print(Xs)
X1 = matricize(Xs, n=1)
print(X1)
X2 = matricize(Xs, n=2)
print(X1)
X3 = matricize(Xs, n=3)
print(X3)
"""

"""
# test factorMatrics2Tensor
a = np.arange(6).reshape(3,2)
b = np.arange(8).reshape(4,2)
c = np.arange(10).reshape(5,2)
print(a)
print(b)
print(c)
r2 = np.einsum('il,jl,kl->ijk', a, b, c)
print(r2, r2.shape) # shape=(3,4,5)
"""
    
    

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:27:28 2024

@author: Jessica
"""
import numpy as np
from scipy import sparse as sp
from functools import reduce

def normalizeWF(psi,**kwargs):
    shape, dtype = psi.shape, psi.dtype
    if np.array_equal(psi, np.zeros(shape, dtype = dtype)) == True:
        NWF = psi
    else:
        NWF = psi/(np.sqrt(np.vdot(psi, psi)))
    return NWF

def sparseMatrices(a, **kwargs):
    return sp.csc_matrix(a)
      
def tensorOperators(matrix2D, **kwargs):
    return reduce(sp.kron, (sp.eye(2**kwargs['a']), matrix2D , sp.eye(2**kwargs['b'])))

def initialVals(params, **kwargs):
    n = params['n']
    N = params['N']
    alpha = params['alpha']
    op = params['opH']
    pulses = params['pulses']
    r = list(np.random.randint(low = 1,high=30,size=N))
    R = [np.power(1/x, alpha) for x in r]
    # r = np.random.random_sample(size = 2**N)
    psi0 = np.random.randn(2**N)
    psi_nm = normalizeWF(psi0)
    return n, N, r, op, pulses, psi_nm, R, alpha

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:49:51 2024

Testing

@author: Jessica
"""

from Part_2A import *
import numpy as np
from numpy import linalg
from scipy import linalg as splinalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy import sparse as sp
import scipy.sparse.linalg
from functools import reduce
import itertools
from scipy import linalg
from scipy.linalg import expm

params = {
'N' : 1,
'tau_list':[1, 0.5, 0.1, 0.05],
'tau': 0.1,
'n': 2,
'alpha': 1,
'T': 10,
'R':[],
'r':[],
'psi_nm':[],
'opH': [X, Y], # Need to change this specific to Model
'pulses': [I, Z] # Need to change this specific to Model
}

# print(r)
# print(R)
mss=10

# print(params['tau_list'])

for i in range(1, 8, 1):
    params['N'] = i
    params['pulses'] = [I, Z, X, Y]
    params['opH'] = [Y, X, Z]
    # params['pulses'] = [I, Z]
    # params['opH'] = [Y, X]
    params['n'] = len(params['pulses'])
    params['alpha'] = 6
    n, N, r, op, pulses, psi_nm, R, alpha = initialVals(params)
    params['r'], params['R'], params['psi_nm'] = r, R, psi_nm

    mss=10

    # print(params['tau_list'])
    
    plt.figure(figsize=[7,5])
    for tau in params['tau_list']:
        params['tau'] = tau
        F = []
        uOp, t = TimeEvolOpForTFH(params, TFH = TogglingFrame_Ising(params)+TogglingFrameH(params))   
        psi_t = [normalizeWF(u@psi_nm) for i,u in enumerate(uOp)]
        F = [1-np.power(np.abs(np.vdot(psi_nm, pt)), 2) for pt in psi_t]
    #     print(F)
        plt.plot(t, F, "--o", label = f"N={params['N']}, τ={tau}, α={params['alpha']}", ms=mss)
        mss -=2
        plt.yscale("log")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Fidelity")
        plt.title("Mitigating the noise with Pulse Sequences different τ")
        plt.grid('on')
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:48:00 2024

Module for Generating Walsh Pulse Sequence 
for Dynamical Decoupling in Long-range interactions

@author: Jessica
"""
import numpy as np
from numpy import linalg
from scipy import linalg as splinalg
import matplotlib.pyplot as plt
from scipy import sparse as sp
import scipy.sparse.linalg
from functools import reduce
import itertools
from scipy import linalg
from scipy.linalg import expm

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
I = np.array([[1, 0], [0, 1]])
H = np.array([[1, 1], [1, -1]])

def WF_Conditions(tupleprdt, **kwargs): # tupleprdt is a list
    for i, tprdt in enumerate(tupleprdt):
        if tprdt[0] == tprdt[1] == 1:
            tupleprdt[i] = I
        elif tprdt[0] == -tprdt[1] == 1:
            tupleprdt[i] = X
        elif -tprdt[0] == tprdt[1] == 1:
            tupleprdt[i] = Y
        elif tprdt[0] == tprdt[1] == -1:
            tupleprdt[i] = Z
    return tupleprdt   
# print(WF_Conditions(tupleprdt = [(1,1), (1,-1)]))

def WF_Generate(params, **kwargs):
    N, lst, W_x, W_y, tupleprdt, q, lst = params['N'], [H], kwargs['W_x'], kwargs['W_y'], [], 0, []
    H0, H1 = np.array([1]), H 
    q = int(np.ceil(np.log2(max(W_x, W_y)+1)))
    if q == 0:
        lst = [H0]
        tupleprdt.append((1, 1)) # Since if the max(wx, wy) is 0, then both are zero.
    else:
        lst = [H1]
        for i in range(q-1):
            lst.append(lst[i])
    Hf = reduce(np.kron, lst)
    w_x, w_y = Hf[W_x-1], Hf[W_y-1]
    Hf = reduce(np.kron, lst)
    wfx, wfy = Hf[W_x], Hf[W_y]
    for i, h in enumerate(wfx):
        tupleprdt.append((h, wfy[i]))
    tupleprdt = WF_Conditions(tupleprdt)
    return tupleprdt
# print(WF_Generate(params, W_x = 3, W_y = 3))
    
def WF_WIList(params, **kwargs):
    W_x, W_y, tupleprdt, ps, Pseq = kwargs['W_x'], kwargs['W_y'], [], [], []
    for i, w_x in enumerate(W_x):
        tupleprdt.append(WF_Generate(params, W_x = w_x, W_y = W_y[i]))
    ps = [[] for _ in range(len(max(tupleprdt,key=len)))]
    padded_tupleprdt = list(zip(*itertools.zip_longest(*tupleprdt, fillvalue=I)))
    for i, p in enumerate(ps):
        for j, padded_ps in enumerate(padded_tupleprdt):
            ps[i].append(padded_ps[i])
    for i, p in enumerate(ps):
        Pseq += [reduce(np.kron, p)]
    return Pseq
# print(len(WF_WIList(params, W_x = [1, 2, 3], W_y =  [1, 2, 3])))
# print(WF_WIList(params, W_x = [1, 2, 3], W_y = [1, 2, 3]))

def WPSresource_Hamiltonian_TimeEvolOp_IsingType(params, **kwargs):
    N, H_r, unitary_timeOp, opH = params['N'], np.zeros((2**params['N'], 2**params['N']), dtype = complex), 0, params['opH']
    R, r, alpha, lst = params['R'], params['r'], params['alpha'], [I for _ in range(N)]
    for op in opH:
        for i in range(N):
            for j in range(i+1, N, 1):
                lst[i] = op
                lst[j] = op
                H_r += R[i]*reduce(np.kron, lst)/(np.power(np.abs(i-j), alpha))
                lst = [I for _ in range(N)]
    tau = params['tau']
    unitary_timeOp = expm(-1j*tau/(params['n'])*H_r)
    return H_r, unitary_timeOp

def WPSeq_TimeEvolOp(params, **kwargs):
    Pseq, unitary_timeOp, timeOpPHrP = params['pulses'], [], np.eye(2**(params['N']))
    params['n'], tau_n_list = len(Pseq), []
#     print(params['tau'])
    Hr, expH_r = WPSresource_Hamiltonian_TimeEvolOp_IsingType(params)
    for k, p in enumerate(Pseq):
        timeOpPHrP = np.linalg.inv(p)@expH_r@p@timeOpPHrP
    t_list = np.arange(0, params['T'], params['tau'])
    unitary_timeOp = [np.linalg.matrix_power(timeOpPHrP, i) for i, t in enumerate(t_list)]
    return unitary_timeOp, t_list

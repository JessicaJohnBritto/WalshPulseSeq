# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:08:54 2024

@author: Jessica
"""

from Part_2A import normalizeWF, initialVals
from Module_WalshPSeq import WF_WIList


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

params = {
'N' : 1,
'tau_list':[1, 0.5, 0.1, 0.05],
'tau': 0.1,
'n': 2,
'alpha': 1,
'T': 10,
    'R': [],
    'r': [],
    'alpha': 1,
'opH': [X, Y], # Need to change this specific to Model
'pulses': [I, Z] # Need to change this specific to Model
}

H = np.array([[1, 1], [1, -1]])

def WF_PulseRotErr(params, **kwargs): 
    """
    **kwargs : W_e, required
        'W_e' is a list of Walsh Indices
    Returns: A list of columns from the
        generalized Hadamard Matrix
        corresponding to each index in
        W_e.
    """
    N, W_e = params['N'], kwargs['W_e']
    H0, H1, lst, q = np.array([1]), H, [], int(np.ceil(np.log2(np.max(W_e)+1)))
    lst = [H1]
    for i in range(q-1):
        lst.append(lst[i])
    Hf = reduce(np.kron, lst)
    Wf_Qbit = [Hf[we] for we in W_e]
    return Wf_Qbit
# print(WF_PulseRotErr(params, W_e = [0, 1, 2, 3]))
        

# params['deltaErr_list'] = list(range(0, params['N'], 2))
params['deltaErr_list'] = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0055, 0.0006, 0.0065, 0.0007, 0.0008]
# W_x = [0, 1]
# W_y = [0, 1]
# Pseq = WF_WIList(params, W_x = W_x, W_y = W_y)
def pulse_rotationErrors(params, **kwargs): # eq21 from the paper
    """
    **kwargs : W_e, W_x, W_y required
        'W_f' is a list of Walsh Indices
    Returns: A list of sublists of PulseErrors stored corresponding to l'th index.
    """
    W_x, W_y, deltaErr_list, W_e = kwargs['W_x'], kwargs['W_y'], params['deltaErr_list'], kwargs['W_e']
    Pseq = WF_WIList(params, W_x = W_x, W_y = W_y)
    Pseq_Err, Wf_Qbit = [[] for _ in range(len(Pseq))], WF_PulseRotErr(params, W_e = W_e)
#     print(len(Pseq))
    for i, ps_err in enumerate(Pseq_Err):
        for j, si in enumerate(Wf_Qbit):
#             print(f'i,j = {i,j}',si[i])
            Pseq_Err[i] += [expm(-1j*(si[i])*(np.pi + deltaErr_list[j])*Pseq[j]/2)]
#             Pseq_Err[i] += [expm(-1j*(si[i])*(np.pi + deltaErr_list[j])*Pseq[j]/2)] # Stored l-wise
#             print(f'i,j={i,j}',Pseq_Err[i][j])
#             Pseq_Err[i] += [expm(-1j*(Wf_Qbit[i][j])*(np.pi + deltaErr_list[i])*Pseq[i]/2)] # Stored sitewise
    return Pseq_Err, Pseq
# print(pulse_rotationErrors(params, W_x = [0, 1], W_y = [0, 1], W_e = [0, 1]))

def WF_TimeEvolOp_HRA_Err(params, **kwargs): # Equations 23
    W_x, W_y, W_e, delErr_lst = kwargs['W_x'], kwargs['W_y'], kwargs['W_e'], params['deltaErr_list']
    params['N'] = len(W_x)
    N, R = params['N'], params['R']
    Wf_x, Wf_y, Wf_e = WF_PulseRotErr(params, W_e = W_x), WF_PulseRotErr(params, W_e = W_y), WF_PulseRotErr(params, W_e = W_e)
    H_RAErr = np.zeros((2**N, 2**N), dtype = complex)
    Pseq_Err, Pseq = pulse_rotationErrors(params, W_x = W_x, W_y = W_y, W_e = W_e)
    params['n'], params['pulses'] = len(Pseq_Err), Pseq
    n, N, r, op, pulses, psi_nm, R, alpha = initialVals(params)
    params['R'], params['r'] = R, r
#     print(Pseq[0].shape)
#     print(Pseq)
    timeOp_PHerrP, timeOp_HRAErr = np.ones((2**N, 2**N)), np.ones((2**N, 2**N))
    unitary_timeOp, pulses = [], np.zeros((2**N, 2**N))
#     lst_Y, lst_X, lst_Z = [Y for _ in range(N)], [X for _ in range(N)], [Z for _ in range(N)]
#     Y_N, X_N, Z_N = reduce(np.kron, lst_Y), reduce(np.kron, lst_X), reduce(np.kron, lst_Z)  
#     HXij, HYij = np.zeros((2**N, 2**N), dtype = complex), np.zeros((2**N, 2**N), dtype = complex)
#     HXji, HYji = np.zeros((2**N, 2**N), dtype = complex), np.zeros((2**N, 2**N), dtype = complex)
    HXij, HYij = np.zeros((2, 2), dtype = complex), np.zeros((2, 2), dtype = complex)
    HXji, HYji = np.zeros((2, 2), dtype = complex), np.zeros((2, 2), dtype = complex)
#     print(f'HYji.shape = {HYji.shape}')
    lst1, lst2 = [I for _ in range(N)], [I for _ in range(N)]
    Hra_lst = []
#     print(Y_N.shape)
#     print(params['N'])
#     print(Wf_e[0][1])
#     print(H_RAErr.shape)
    for l in range(len(W_e)):
        for op in params['opH']:
            for i in range(params['N']):
                for j in range(i+1, params['N'], 1):
                    for k in range(params['n']):
#                         print('Blablabla....................Blablabla....................Blablabla....................')
                        if np.array_equal(op, X) == True:
#                             print('Blablabla....................Blablabla....................Blablabla....................')
                            HXij += ((1-Wf_x[i][k])/2)*Wf_x[j][k]*((((1-Wf_y[i][k])/2)*Y) - (((1+Wf_y[i][k])/2)*Z))
                            HXji += ((1-Wf_x[j][k])/2)*Wf_x[i][k]*((((1-Wf_y[j][k])/2)*Y) - (((1+Wf_y[j][k])/2)*Z))
#                             print(f'HXij.shape = {HXij.shape}')
                        elif np.array_equal(op, Y) == True:
#                             print(f'HYji.shape = {HYji.shape}')
                            HYij += ((1-Wf_y[i][k])/2)*Wf_y[j][k]*((((1+Wf_x[i][k])/2)*Z) - (((1-Wf_x[i][k])/2)*X))
                            HYji += ((1-Wf_y[j][k])/2)*Wf_y[i][k]*((((1+Wf_x[j][k])/2)*Z) - (((1-Wf_x[j][k])/2)*X))
                    HXij, HYij, HXji, HYji  = HXij/params['n'], HYij/params['n'], HXji/params['n'], HYji/params['n']
                    if np.array_equal(op, X)== True:
                        lst1[i], lst1[j] = HXij, op
                        lst2[i], lst2[j] = op, HXji
#                         print(f'reduce(np.kron, lst1).shape = {reduce(np.kron, lst1).shape}')
                        H_RAErr += (R[i]/(np.power(np.abs(i-j), alpha)))*(delErr_lst[i]*reduce(np.kron, lst1) + 
                                                                          delErr_lst[j]*reduce(np.kron, lst2))

#                         H_RAErr += (R[i]/(np.power(np.abs(i-j), alpha)))*(delErr_lst[i]*Wf_e[i][l]*reduce(np.kron, lst1) + 
#                                                                           delErr_lst[j]*Wf_e[j][l]*reduce(np.kron, lst2))
                    elif np.array_equal(op, Y)== True:
                        lst1[i], lst1[j] = HYij, op
                        lst2[i], lst2[j] = op, HYji
                        H_RAErr += (R[i]/(np.power(np.abs(i-j), alpha)))*(delErr_lst[i]*reduce(np.kron, lst1) + 
                                                                          delErr_lst[j]*reduce(np.kron, lst2))
#                         H_RAErr += (R[i]/(np.power(np.abs(i-j), alpha)))*(delErr_lst[i]*Wf_e[i][l]*reduce(np.kron, lst1) + 
#                                                                           delErr_lst[j]*Wf_e[j][l]*reduce(np.kron, lst2))
                    lst1, lst2 = [I for _ in range(params['N'])], [I for _ in range(params['N'])]
                    HXij, HYij = np.zeros((2, 2), dtype = complex), np.zeros((2, 2), dtype = complex)
                    HXji, HYji = np.zeros((2, 2), dtype = complex), np.zeros((2, 2), dtype = complex)
#         print(H_RAErr.dtype)
        Hra_lst +=[H_RAErr]
        timeOp_HRAErr = expm(-1j*H_RAErr*params['tau']/params['n'])
#         print(f'Pseq_Err[l][0].shape = {Pseq_Err[l][0].shape}')
        pulses = reduce(np.matmul, Pseq_Err[l])
#         print(f'pulses.shape = {pulses.shape}')
#         print(f'timeOp_HRAErr.shape = {timeOp_HRAErr.shape}')
        timeOp_PHerrP = np.linalg.inv(pulses) @ timeOp_HRAErr @ pulses @ timeOp_PHerrP
    t_list = np.arange(0, params['T'], params['tau']) 
    unitary_timeOp = [np.linalg.matrix_power(timeOp_PHerrP, i) for i, t in enumerate(t_list)]
#     print(f'sum={sum(Hra_lst)}')
    return unitary_timeOp, t_list

WF_TimeEvolOp_HRA_Err(params, W_x =[0,1], W_y = [0,1], W_e = [0,1])
            
   


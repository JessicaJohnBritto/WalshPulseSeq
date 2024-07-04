# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:11:15 2024

@author: Jessica
"""
from Section_2.Part_2A import normalizeWF, initialVals
from Section_3.Module_RotationAngleErrorsDoubleAveraging import *
from Section_2.Module_WalshPSeq import WF_WIList

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
'pulses': [] # Need to change this specific to Model
}
max_index = 8
W_x, W_y = list(range(0, max_index, 1)), list(range(0, max_index, 1))
params['deltaErr_list'] = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0055, 0.0006, 0.0065, 0.0007, 0.0008]
params['alpha_list'] = [0.05, 0.5, 1, 1.2, 2, 3, 6, 12]
for alpha in params['alpha_list']:
    params['alpha'] = alpha
    mss = 10

    plt.figure(figsize=[7,5])
    for tau in params['tau_list']:
        params['tau'] = tau
        F = []
        uOp, t = WF_TimeEvolOp_HRA_Err(params, W_x = W_x, W_y = W_y, W_e = W_x)
        n, N, r, op, pulses, psi_nm, R, alpha = initialVals(params)
        alphaf = params['alpha']
#         t = np.arange(0, params['T'], params['tau'])
#         print(f'alpha = {alphaf}')
#         print(f'tau = {tau}')
#         print(f'psi_nm.shape = {psi_nm.shape}')
#         print(f'length of t = {len(t)}')
#         uOp = [np.linalg.matrix_power(timeOpPHrP, i) for i, _ in enumerate(t)]
#         print(f'length of uOp = {len(uOp)}')
#         print('*************************************')
        psi_t = [normalizeWF(u@psi_nm) for i,u in enumerate(uOp)]
        F = [np.power(np.abs(np.vdot(psi_nm, pt)), 2) for pt in psi_t]
    #     print(F)
        plt.plot(t, F, "--o", label = f"N={params['N']}, τ={tau}", ms=mss)
        mss -=2
#         plt.yscale("log")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Fidelity")
        plt.title(f"Mitigating the noise with Pulse Sequences different τ, α={params['alpha']}")
        plt.grid('on')
    plt.show()

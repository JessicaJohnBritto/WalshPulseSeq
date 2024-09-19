# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:27:28 2024

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

# Pauli Matrices
sigmaZ = sp.csc_matrix([[1, 0], [0, -1]])
sigmaX = sp.csc_matrix([[0, 1], [1, 0]])
sigmaY = sp.csc_matrix([[0, -1j], [1j, 0]])
sigmaI = sp.csc_matrix([[1, 0], [0, 1]])
sigmaH = sp.csc_matrix([[1, 1], [1, -1]])

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
I = np.array([[1, 0], [0, 1]])

H = np.array([[1, 1], [1, -1]])

def normalizeWF(psi,**kwargs):
    '''
    Returns a normalized wavefunction.
    Args: psi - a column vector.
    '''
    shape, dtype = psi.shape, psi.dtype
    if np.array_equal(psi, np.zeros(shape, dtype = dtype)) == True:
        NWF = psi
    else:
        NWF = psi/(np.sqrt(np.vdot(psi, psi)))
    return NWF

def sparseMatrices(a, **kwargs):
    '''
    Generates sparse matrices for a given dense matrix.
    Args: a - a 2D numpy array
    '''
    return sp.csc_matrix(a)
      
def tensorOperators(matrix2D, **kwargs):
    '''
    Returns tensor product of an operator acting on specific qubits  on the system.
    Args: matrix2D - a 2X2 dim numpy array.
    kwargs: a - no. of sites to the left of the matrix2D,
    b - no.of sites to the right of the matrix2D.
    '''
    return reduce(sp.kron, (sp.eye(2**kwargs['a']), matrix2D , sp.eye(2**kwargs['b'])))

def initialVals(params, **kwargs):
    '''
    Initializes initial_wavefunction and it's normalized form based on number of qubits.
    Returns:
    n: length of the pulse sequence,
    N: total number of qubits,
    r: coupling constants generated randomly for N qubits,
    op: params['opH'],
    pulses: params['pulses'],
    psi_nmn: normalized initial wavefunction randomly generated from Gaussian Distribution,
    R: inverse of r,
    alpha: extent to which the qubits can interact,
    Args: params: dictionary
    '''
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

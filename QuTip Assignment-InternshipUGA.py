{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "3a3a8c88",
   "metadata": {},
   "source": [
    "# QuTip Practice Codes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9aa707eb",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\anacondafinal\\lib\\site-packages\\qutip\\__init__.py:66: UserWarning: The new version of Cython, (>= 3.0.0) is not supported.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "import qutip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c81d7479",
   "metadata": {},
   "outputs": [],
   "source": [
    "from qutip import *\n",
    "import numpy as np\n",
    "from numpy import linalg\n",
    "from scipy import linalg as splinalg\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import normalize\n",
    "from scipy import sparse as sp\n",
    "import scipy.sparse.linalg\n",
    "from functools import reduce\n",
    "from sklearn.preprocessing import normalize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b3abd5cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# We make use of Quantum Object Class using matrix representation\n",
    "# since we need a data structure to store the properties of a \n",
    "# quantum operator and its eigenstates and values.\n",
    "# The output corresponds to a 1X1 matrix with one zero entry.\n",
    "# Names of classes are capitalized unlike the functions in Python\n",
    "print(Qobj(), '\\n \\n********************** \\n')\n",
    "\n",
    "# Creating a user-defined data set by passing them in the form of\n",
    "# arrays into the Q. Object class\n",
    "print(Qobj([[1], [2], [3], [4], [5]]), '\\n \\n********************** \\n') # 5X1 matrix - column vector\n",
    "\n",
    "x = np.array([[1, 2, 3, 4, 5]])\n",
    "print(Qobj(x), '\\n \\n********************** \\n')\n",
    "\n",
    "r = np.random.rand(4, 4)\n",
    "print(Qobj(r), '\\n \\n********************** \\n')\n",
    "\n",
    "# Although dims and shape appear to be the same,\n",
    "# dims keep track of the shapes of the individual \n",
    "# components of a multipartite system - check tensor section\n",
    "\n",
    "# QuTip has built-in functions of commonly used state vectors\n",
    "# Fock state ket vector: N = no. of levels in Hilbert space, #m = level containing excitation\n",
    "N = 4\n",
    "m = 0\n",
    "alpha = 4+1j\n",
    "f = fock(4, 3)\n",
    "print(f, '\\n \\n********************** \\n')\n",
    "\n",
    "#Empty ket vector\n",
    "zero_ket(N)\n",
    "\n",
    "#Fock density matrix (outer product of basis) - hermitian by default\n",
    "# Arguments same as fock(N,m)\n",
    "fock_dm(N,3)\n",
    "\n",
    "#Coherent state, alpha = complex no. (eigenvalue)\n",
    "coherent(N,alpha)\n",
    "\n",
    "#Coherent density matrix (outer product)\n",
    "coherent_dm(N,alpha)\n",
    "\n",
    "#Thermal density matrix (for n particles), n = particle number expectation value \n",
    "print(thermal_dm(N,1010)) # What does this mean by particle number expectation value?\n",
    "\n",
    "print(coherent_dm(N,alpha).dims) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "162d3b27",
   "metadata": {},
   "outputs": [],
   "source": [
    "q = destroy(4)\n",
    "print(q)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "90ed46a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = sigmax()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a44a38b",
   "metadata": {},
   "outputs": [],
   "source": [
    "q + 5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c25949c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "x * x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "171d057e",
   "metadata": {},
   "outputs": [],
   "source": [
    "q ** 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6052c032",
   "metadata": {},
   "outputs": [],
   "source": [
    "x / np.sqrt(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "500aea06",
   "metadata": {},
   "outputs": [],
   "source": [
    "vac = basis(5, 0)\n",
    "print(vac)\n",
    "a = destroy(5)\n",
    "print(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d1d9598",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(a.dag())\n",
    "print(a.dag() * vac)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c38f52e",
   "metadata": {},
   "outputs": [],
   "source": [
    "c = create(5)\n",
    "print(c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4268dac",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(c * vac)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f57dc575",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(c * c * vac)\n",
    "print(c ** 2 * vac)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8ce28ee5",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(c * a * vac) # Applying number operator on |0>\n",
    "print(c*a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0a6d5933",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(c * a * (c * vac)) # Applying number operator on |1>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c6441e32",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(c * a * (c**2 * vac).unit()) # Applying number operator on |2>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "beaccb9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "ket = basis(5, 2)\n",
    "print(ket)\n",
    "n = num(5)\n",
    "print(n)\n",
    "print(n * ket)\n",
    "ket = (basis(5, 0) + basis(5, 1)).unit()\n",
    "print(ket)\n",
    "print(n * ket)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "621492a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "ket = basis(5, 2)\n",
    "print(ket * ket.dag())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6fe223e3",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "print(fock_dm(5, 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0012fef8",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(ket2dm(ket))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "373c95cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = coherent_dm(5, 1.25)\n",
    "\n",
    "y = coherent_dm(5, np.complex(0, 1.25))  # <-- note the 'j'\n",
    "\n",
    "z = thermal_dm(5, 0.125)\n",
    "\n",
    "print(np.testing.assert_almost_equal(fidelity(x, x), 1))\n",
    "\n",
    "np.testing.assert_almost_equal(hellinger_dist(x, y), 1.3819080728932833)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95721d14",
   "metadata": {},
   "outputs": [],
   "source": [
    "vac = basis(5, 0)\n",
    "print(vac)\n",
    "\n",
    "one = basis(5, 1)\n",
    "print(one)\n",
    "\n",
    "c = create(5)\n",
    "print(c)\n",
    "\n",
    "N = num(5)\n",
    "print(N)\n",
    "\n",
    "np.testing.assert_almost_equal(expect(N, vac), 0)\n",
    "\n",
    "np.testing.assert_almost_equal(expect(N, one), 1)\n",
    "\n",
    "coh = coherent_dm(5, 1.0j)\n",
    "\n",
    "np.testing.assert_almost_equal(expect(N, coh), 0.9970555745806597)\n",
    "\n",
    "cat = (basis(5, 4) + 1.0j * basis(5, 3)).unit()\n",
    "\n",
    "np.testing.assert_almost_equal(expect(c, cat), 0.9999999999999998j)\n",
    "print(expect(N, (c**2 * vac).unit()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4497d994",
   "metadata": {},
   "outputs": [],
   "source": [
    "Sxi = sigmax()\n",
    "Sxi1 = sigmax()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "30e3137c",
   "metadata": {},
   "source": [
    "# Using Numpy and Scipy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9dacb856",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.+0.j 0.+0.j 0.+0.j 0.+0.j]\n",
      " [0.+0.j 0.+0.j 4.+0.j 0.+0.j]\n",
      " [0.+0.j 4.+0.j 0.+0.j 0.+0.j]\n",
      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]\n"
     ]
    }
   ],
   "source": [
    "## Defining XY Hamiltonian 1D - nearest neighbor interactions only\n",
    "sx = np.array([[0,1], [1, 0]])\n",
    "sy = np.array([[0, -1j], [1j, 0]])\n",
    "iden = np.eye(2)\n",
    "sx1sx2 = np.matmul(np.kron(sx, iden),np.kron(iden, sx))\n",
    "sy1sy2 = np.matmul(np.kron(sy, iden),np.kron(iden, sy))\n",
    "H = sx1sx2 + sy1sy2\n",
    "# print(H)\n",
    "N = 2\n",
    "Hm = np.zeros((2**N, 2**N), dtype = np.complex_)\n",
    "Sxisxi1 = np.zeros((2**N, 2**N), dtype = np.complex_)\n",
    "Syisyi1 = np.zeros((2**N, 2**N), dtype = np.complex_)\n",
    "for i in range(N):\n",
    "    Sxisxi1 = np.matmul(np.kron(sx, np.eye(2**(N-1))), np.kron(np.eye(2**(N-1)), sx))\n",
    "    Syisyi1 = np.matmul(np.kron(sy, np.eye(2**(N-1))), np.kron(np.eye(2**(N-1)), sy))\n",
    "    Hm += Sxisxi1 + Syisyi1  \n",
    "\n",
    "# normed_matrix = normalize(Hm, axis=1, norm='l1') - applicable only for real entry matrices\n",
    "print(Hm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "68ed2c35",
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy import linalg\n",
    "from scipy.linalg import expm, sinm, cosm\n",
    "\n",
    "# Pauli Matrices\n",
    "sigmaZ = sp.csc_matrix([[1, 0], [0, -1]])\n",
    "sigmaX = sp.csc_matrix([[0, 1], [1, 0]])\n",
    "sigmaY = sp.csc_matrix([[0, -1j], [1j, 0]])\n",
    "sigmaI = sp.csc_matrix([[1, 0], [0, 1]])\n",
    "sigmaH = sp.csc_matrix([[1, 1], [1, -1]])\n",
    "\n",
    "Z = np.array([[1, 0], [0, -1]])\n",
    "X = np.array([[0, 1], [1, 0]])\n",
    "Y = np.array([[0, -1j], [1j, 0]])\n",
    "I = np.array([[1, 0], [0, 1]])\n",
    "\n",
    "H = np.array([[1, 1], [1, -1]])\n",
    "\n",
    "params = {\n",
    "'N' : 1,\n",
    "'tau_list':[1, 0.5, 0.1, 0.05],\n",
    "'tau': 0.1,\n",
    "'n': 2,\n",
    "'alpha': 1,\n",
    "'T': 10,\n",
    "'opH': [X, Y], # Need to change this specific to Model\n",
    "'pulses': [I, Z] # Need to change this specific to Model\n",
    "}\n",
    "\n",
    "def normalizeWF(psi,**kwargs):\n",
    "    shape, dtype = psi.shape, psi.dtype\n",
    "    if np.array_equal(psi, np.zeros(shape, dtype = dtype)) == True:\n",
    "        NWF = psi\n",
    "    else:\n",
    "        NWF = psi/(np.sqrt(np.vdot(psi, psi)))\n",
    "    return NWF\n",
    "\n",
    "def sparseMatrices(a, **kwargs):\n",
    "    return sp.csc_matrix(a)\n",
    "      \n",
    "def tensorOperators(matrix2D, **kwargs):\n",
    "    return reduce(sp.kron, (sp.eye(2**kwargs['a']), matrix2D , sp.eye(2**kwargs['b'])))\n",
    "\n",
    "def initialVals(params, **kwargs):\n",
    "    n = params['n']\n",
    "    N = params['N']\n",
    "    alpha = params['alpha']\n",
    "    op = params['opH']\n",
    "    pulses = params['pulses']\n",
    "    r = list(np.random.randint(low = 1,high=30,size=N))\n",
    "    R = [np.power(1/x, alpha) for x in r]\n",
    "    # r = np.random.random_sample(size = 2**N)\n",
    "    psi0 = np.random.randn(2**N)\n",
    "    psi_nm = normalizeWF(psi0)\n",
    "    return n, N, r, op, pulses, psi_nm, R, alpha\n",
    "n, N, r, op, pulses, psi_nm, R, alpha = initialVals(params)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "64b23572",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1, 0.5, 0.1, 0.05]\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAdgAAAFNCAYAAACjcn5pAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAAsTAAALEwEAmpwYAAEAAElEQVR4nOxdd5wURdp+avLsbN5ll4UFFpYgOSOCIAoICpgzd6bTO9PpneE77wyHl/ROvWi4O/WMmLOCgoIYUBAVJKeFBZZdWDanSd1d3x/VNdsz03FmlkWdxx8/d7qqq6u7qru63/d93odQSpFGGmmkkUYaaaQWtu7uQBpppJFGGml8H5FeYNNII4000kijC5BeYNNII4000kijC5BeYNNII4000kijC5BeYNNII4000kijC5BeYNNII4000kijC5BeYL/nIIT8mxByl075bwghj3fRsacRQnZ0Rdsax6OEkIFH63gafTiq5xxz7IWEkOU65TMIIVVHoR9H5ThpxIMQsooQcpX8d9R8IIRMJYTsIoS0EULOIoQUE0I+IYS0EkIe7L5ef3+RXmC/oyCEVBJCQoSQwpjt6+WFpgwAKKXXUEp/L5fFPfgopX+ilF6Voj5FLXCU0k8ppUNS0bbKsSIPkmMJXXnOJo69mFJ6Kv+d7AuHfI0D8gO5jhDyOiGkJDW9Tag/vyGE7JX7U0UIeam7+vJdQOx8APA7AA9RSjMppW8C+CmAOgDZlNJbjmbfCCGXE0I+O5rH7A6kF9jvNvYCuJj/IISMBJDRfd1J43uIGyilmQAGA8gF8Lfu6AQh5DIAPwYwS+7PBAAruqMv32H0A7Al5vdWmkC2IUKII2W9+j6DUpr+9x38B6ASwJ0A1im2PQDgDgAUQJm87SkAfwDgA+AHIAFok//1ArAIwHOKNi4FsA9APYC75OPMkssmAfgCQBOAGgAPAXDJZZ/Ix22X274QwAwAVTF9vhXARgDNAF4C4FGU/5/cbjWAq+T2Bqqc+x8BiAAC8rEekrdTANcA2CX38WEARLHflQC2AWgEsAxAP41rWya3dRmA/WBv+Xcoyt0A/i73s1r+2y2XxZ7zrwAcBNAKYAeAmfJ2G4DbAVTI1/plAPka/fkYwLny31Plvs2Tf88EsEH++3IAnxmNB4BbANTK1/oKnTm2CsBVit/XA9isuNYDFWVPAfhDF16DhwD8XaevOQCekM/pINict8tldrB7ow7AHvk8KACHYl7OUrS1CNH3xGQAn4PNqW8BzIi5Rr8HsFo+v+UAChXlJyr2PQDgcsUcegBsfh0G8G8AXrmsEMC78j4NAD4FYNM479kAtoPdTw/Jc+UqlflQAXbv++X58AKAMICQ/HuW3nig8574idznT4zuKWjcjwCGgt27onzsJpXzmobO55TyXn+vu5+9Vv6lv2C/21gDIJsQMpQQYgdwEYDn1CpSStsBnAagmjITUSaltFpZhxAyDMAjABYCKAF7aPVWVBEB/BLsAXAC2MP9Orn96XKd0XLbWua7CwDMBdAfwCiwhwAIIXMB3Ax2ow8Ee0irglJ6B9hD5wb5WDcoiucDmCi3fQGAOXL7ZwL4DYBzAPSQ939B6xgyTgQwRD7PuwkhQ+Xtd4A9dMcAGA324nFn7M6EkCEAbgAwkVKaJfelUi7+OYCzAJwE9qLTCPYAUsPH6LweJ4EtEtMVvz+O3UFnPHqic1x/AuBhQkiexnGV51II4FwA643qxuyXqmuwBsClhJDbCCET5PmuxFMABLC5MxbAqWAvaQBwNdi8GAv25Xuehf73BrAEbMHOB3tBfI0Q0kNR7RIAVwAoAuCS64AQ0g/AewD+BTbnxgDYIO9zH5hVYIzc594A7pbLbgF7EeoBoBhs3sZ9Zcpj8jrY3CsEWxinqp0HpbQcbGFcIM+HiwEsBvAX+feHMDceJ4EtkHNM3lNx9yOldBvYwvuFfOxclf5+yp9TYC/118i/T1M7v2MV6QX2u49nwb46Z4O9SR5Moq3zALxDKf2MUhoCu+EjNzal9GtK6RpKqUAprQTwH7Abzgr+SSmtppQ2AHgH7AEDsJvvSUrpFkppB9hXRCK4j1LaRCndD+AjRfvXALiXUrqNUioA+BOAMfJDUAv3UEr9lNJvwb5cRsvbFwL4HaW0llJ6BMA9YObLWIhgXyrDCCFOSmklpbRC0Z87KKVVlNKgfL7naZjePkbndZ4O4F7Fb9UFVgdhue9hSulSsK8CPZ/xPwkhTWDnXwP2EmQFKbkGlNLnwBaAOWDnW0sI+RUAEEKKAZwO4BeU0nZKaS2YKfsiefcLwL5+D8jz7l4L/f8RgKWU0qWUUolS+gGAr+TjcTxJKd1JKfWDffWNkbdfAuBDSukL8vWup5RuIIQQMP/nLymlDZTSVrD5yPsbBnvB7Sfv9ymVP+ticDqALZTSVymlYTBLyiEL5xYLM+OxSL7Gfpi7p7Tuxx8E0gvsdx/Pgt3IlwN4Jsm2eoGZsQAA8kJXz38TQgYTQt4lhBwihLSA3VCF8c3oQvkA6ACQqXbsmL9T0X4/AP8ghDTJC0YDmLmqN7Sh19d9irJ98rYoUEp3A/gF2IOqlhDyIiGE1+sH4A1Ff7aBLUbFKv34AsBgeSEZAzbOfeQvmElg5mCzqJcfhmrnpYYbKaW5lNLelNKF8guFaaTwGoCyoJ1ZYL7gawD8nhAyR27HCaBG0dZ/wL4ogfi5pRw7I/QDcD5vV277RLAFkENrnvQB+6qMRQ+wWImvFW2+L28HgPsB7AawnBCyhxByu0bfYu9XisTvG8DceByIqW90T2ldmx8E0gvsdxyU0n1gwU6ng5mLdKsblNcAKOU/CCFeAAWK8kfB/D2DKKXZYOYhYrXPZo4N9nDSg9XAjAMAfiYvFvyfl1L6ucV2AOZ3Vb6l95W3xXeS0ucppSfK9SmAPyv6c1pMfzyU0jgLhPyi8zWAm8B8oCEwv97NACoopXUJnEOy6EB0QF1PrYqpuAYx7YUppa+A+fJHyO0EwXyfvJ1sSulweZcaRM+nvjFNtuucywEAz8b00UcpvU+vj4p9y1W214H5Qocr2syRzaGglLZSSm+hlA4AcAaAmwkhM1XaiTov+cvY6L4x6q/ReNCY+oneU1bu3zBS95w5qkgvsN8P/ATAKZT5WfVwGEABISRHo/xVAAsIIVMIIS6wrw7lxM4C0AKgjRByHIBrVdofYLXzMl4GcIXsT84AC7DSg9Vj/RvArwkhwwGAEJJDCDk/sa7iBQB3EkJ6yF+Rd0PF900IGUIIOYUQ4gYL0uBBZrw/f+TmNLmtM3WO+TGYL5Obg1fF/FZDMuNhhA0ALiGE2GX/uaqrIFXXQKZ1zCOEZBFCbISQ0wAMB7CWUloDFlz0ICEkWy4vJ4TwPr0M4EZCSKnsb479ItwA4CJCiJMQEuujfQ7snpgjn6uHMLpbKYyxGMAsQsgFhBAHIaSAEDKGUioBeAzA3wghRfL59Za/xkEImU8IGSgvmM1gX5GSSvtLAAwnhJwjm3FvhM6LjglYnZPJ3FOHAZTKzxkj7AVwnMl2jymkF9jvASilFZTSr0zU2w62OOyRzTq9Ysq3gPm5XgR7O24DizYNylVuBTNHt4I9IGIDmRYBeFpu+wKL5/AegH+C+Wl2gwW1QHHsWPwDzD/USAj5p4n23wD7cnpRNm9vBgv6SgR/APPDbQSwCcA38rZYuMGCWerATGVFAH6t6P/bYGbAVrDzPV7nmB+DveB8ovFbDYuQ4HiYwE0AFoBFhy4E8KZGvVRdgxYwi8l++Zh/AXAtpZRzKS8FCzDaChac8yo6zbiPgUW4fgs2VrGWnrvAvjQbwfzpz/MCSukBADyY5wjYV9ttMPHslP2Op4MFLTWALeTcj/8ryPNcno8fotMXPkj+3QbmHniEUvqRSvt1AM4Hu7718n6rjfqlA0tzMsl7aiUYZegQIcTIAvMAgDMIIdtMtn3MgKj7ztNIAyCEZII9zAZRSvce5WMPBbth3TE+wzTSSAqEJWHZC8CZnltpdCXSX7BpRIEQsoAQkkEI8YG9OW5CJ62iq499NiHELZvx/gwW0Zx+AKaRRhrfSaQX2DRicSY6EygMAnCRBkWgK/AzMJN0BZjfKdbHm0YaaaTxnUHaRJxGGmmkkUYaXYD0F2waaaSRRhppdAHSC2waaaSRRhppdAHSiggmUVhYSMvKypJup729HT6fL/kOfU+Rvj7aSF8bbaSvjT7S10cbyV6br7/+uo5S2kOtLL3AmkRZWRm++sqQamqIVatWYcaMGcl36HuK9PXRRvraaCN9bfSRvj7aSPbaEEI0U2+mTcRppJFGGmmk0QVIL7BppJFGGmmk0QVIL7BppJFGGmmk0QVIL7BppJFGGmmk0QVIL7BppJFGGmmk0QVIL7BppJFGGmmk0QVIL7BppJFGGmmk0QVI82CPcSzZswT/+OYfONR+CD19PXHTuJswb8C8Lt03dr/ppdPxSdUnCfXBLPgxa9prUPJqScJ9TbRveu0kWpboMbJd2SCEoDnYHHX9rV6bVJ1/d7Rj9XipvjapGnOzbaYKXXEMozbNHvNoz41jAelk/yYxYcIEmspEE6EDB3DgmmsRqqyEq6wMff79KFx9+kTVXbJnCRZ9vggBMRDZ5rF7sGjKIsOJmei+yv2KGil+9aqIXvXA4VyAAChqAmoKCKQ/3445Uy81dR5GdVLRV6vXJ7adh5fcjV+81IFe9UB1AfD3CzNw/bzfAYDmMWLLihopbn9VQu8GAnf//lHnqddXtWOoQTkeYp9iDH3i2bhrnQgSuY5qY/pBeGPS46E3V2LLdv/mAty5919Jj38sjOaDsuxwLmADQXFz/JjHtpmKuaqE3vWInSstV1+Paeefb6ods/N23oB5utcqdhFO5PzV+gbA8JljBSlINPE1pXSCall6gTWHVC+wFfPmI7RnD0ApYLPB1b8/ype8G1X31FdPRU17TVwbJb4SLD9vue5xEt1Xud+DjwnoXcf8CHyWEAAiAWp7OHDKJ5tMnYdRnVT01cp+au3c+vcDkXMVCXtQPPALdtNqHSO27MHHBPSqB+wUceep11etY8RCOR4SATwDyuOudSJI5DpWzJuP0N69gCRFzvXaK0JJj4feXIktqym04aafxLdhdfxjYTQflGXK+0Jr/vM2UzFXldC7HrFzRepZgpEfrTTVjtl5u/y85brXSnleiZ6/Wt8AGD5zrKArF9i0ibibENq7l00QAJAkhCor4+ocaj+kuq/W9lTsqyzvVd/ppCeKOnYKFB1hOuhmzsOoTir6amU/tfrKc7VTdu567aiVRRZXIO48U9FXZR9tFKrXOhEk0rdQZSVbXIHIuR5qVw/psHKOocpKzbkSW1Z0RILaI8zq+KvtrzcftO4Lrfmv16dk+hp7XymvR+xcIYcPm27Hyrw1e+8kev6a88HgmXOsIB3k1E1w9evX+cNmg0tFSKCnr6fqvlrbU7Gvsry6oHM7RefbuiR/wQKAU2ma0TgPZ0mJbp1U9NXKfmr1j+R0/uZv4T19PXWPEVtWXQBI/AchUedppR0tHM7t/FsiUL3WiSCR6xh1bPlcUzEeUe3GzJXYMj4Hkzme1v7Kax07H7Tui9gxN9OnZPrqKot+hiivh7KPEgHE4mLNdhzKsphrbtRvvXtHrb5WO1pQmw96c+RYQ3qB7Sb0/uc/2B+EwCX7bmJx07ib4LF7orZ57B7cNO4mw/ZvGncT3Ha35X2Vx/zzeXYAbNGoyQNavKzOoXyC8L23sfN48AHD8yj+7d3sD9mcE1vnpnE3wWVzJdVXK/uptfPYGexaSaTTj3TTuJt0jxFb9ufz7GjIZt80zt69o85TbzzUjqGGf5xpi/RR6FOseq0TQSLXsc+/H4WjiAmIOPv2RZ9/P5qS8Yick8p86vPvRwGnM1IWvve2lIx/LG4adxP+fRZrV20+/P3CDEjovC+aZSEWfh202kx1X3v/7W/sD5Xr8efz7BAIW/yFPsVouu5azXaK77iD/aFyfxr1+6ZxN+GJeezejb1WSiR6/n3+/Shgs0XNhz7/fhS27GwA0HzmHCtIm4i7Ca5evQAA+ZddhuLbf6Vahzv/7117L5pDzSj0FuLWCbeaCoqYN2AeGgON+PO6PwNgvg4zUXu8/K7Vd6E2LwwA2HFiKX477RAu/FjEuZ9ThP51N04bdxEAwCm//RZeey163Phz1TYdubkAgJ73LEKeSqDFvAHzsLNxJ/63+X8J9fX2T2+3tJ9aO62TdgHP/AeLZ9jw1czece3oHYOX2Ut7QfzZNOD+51Hyxz9GBV7MGzAPte21+Os3fzVsJ8eVoxpF3JBdDQBovmgWpvz2X5bO0ej8jc4xFq4+fZB/6aWofeBB9H3sv3D16YN56IOgEMRvv/it6XbU2gWAnDPPRK/77o0vC4fhGTEC/V99BeUAhJIC3PPFPfAL/oTHPxbzBsxDYEoV8OTf8cqJNnw+J2Y+zANsD9+K1UMJXl5Yirv2jwX++zb6PPKwZrCN8r4KS+GU9JVbhvKvvALFt90WuR53rr4TtXkCHBToGDcY459/C6tWrdJsx56VBQDo9ec/I2fBfNV+//nLP6Mx2Ig8Tx5+NfFXke3zBszD/qGfAngLT86yYeNJ8feOsp3ff/F7tAvt6JnRE78Y/wvD83f16QNIErJmz0LpvzrnfMaECWhbuTIlMQhdifQXbDdBCoUAAK0agQcc8wbMw0l9TgIA3D35bks35Am9TgAAjCocheXnLTe977wB8zCjzwycM+gcAMBxG5sAAGfU9QUAzOwxNVJXbGsDALSt/kyzvbDs//Gv36BZpyy7DABwctbJlvvKYWW/WAxzlAIAph3OjWuH/+20OTXL+PGHNWUAUPePjugxAgBwSp9TVNvpl90Pt0+6HZ9d/Bk+vehTbLxsI5aftxx3Tr4Tvzn+N7gqcyEAoGSztj8tUSRyHdvXrAUAiE1NkW0z+80EAJRmliY1Hi1LlmiWBTZvjvytbH/pOUtTRvuYkMPGasp+j+aYT91Gsfy85Sjb0w4AEOrrdducN2AewhJ7aU3m2nDQMGurbUXnM2TegHkQJCHyO+ObnYbthA8eBAAENm9SLZ83YB5+OuqnAICLh1wc1+8RdnbvzKjroXte8wbMQ5GvCADwwvwXLJ1/6wcfRv1uW6n/3DxWkF5guwk2DzOXqH3RxWJa6TQAQN/svpaOUZTBJvOcsjkWewf8dcZfcc+UewAAtvPnY0TBCJC5MwB0vvECgF3+Os1ZcIZmW+7ycgCAb+oUzTqjeowCAPRz99OsowWP3YMrhl9heT8lCkoHAgDcs2aoljtsDlw2/DLVsjE9xuD4kuMBABnHT2btDBoUV4/7m/giFIt3z34XC4cuVC37z7f/wUfiOgBAzhna1zoZlPhKcGb5mabrZ89l88rRo1NrmpvBzxt8XuIdcTqRf4X6eLrKy5E1d27Utp+MUAklThLZPeUv0Vknqpa3ewgqTh0KAMiaNQtATKyBBuaUzUH/nP4p6aMtMxMAkHPOOVHbZ/adiUF5g+AoKkLu+cbj4Bl6HAAgY+JEzTrjiscBAIbkD4krC5bkAwAOjCoyPNZFQ5jly07shnU5bDk5yPvRj6K25V1ySeTZcywjvcCmYYgCTwFemP8Cemf27rJj8BuOdoaMfGdgt9khUcm4YrLHQNceI400tMDvT7V53i+LvfhzK1QanUgvsN0EGgwCAJqXLDWsu66GfbkcbDto6RjVbcxn9/Xhry32jvlK/voV8xc2v/EGAKB1OeOriW3tkXpSaysr+/BDaCFYUQEACGzcqFlnT/MeAEBlsNJyXwNiAC/vfNnyfko01rDj+j9drVouSAJe2/maatnXh7/GukNsjDrWsf+Hqw7E1TvScQQAsPqg+jGueP8KvLHrDdUyO7HD5Wemv9YPPtA4i+RQ016DtyreMl2/7eOPAQBCfUNkW0hiro+le43ntSbCYTS98opqUaiiAq3vvx+1bfG2xYkfSwOth6rY8T5fq1ruC1D0WrUNAND+GRtPobbWsN1llcuwt3lvSvooNjcDANpWrIjavmL/Cuxq3AWhthZNr7xq2E5g+w72/63bNOvsbGSm5m0N8XWcNWz883YYc7nf2K0+v/UgNTej8bnnorY1Pv98lGviWEV6ge0m8AQfQo3xpGwMNgIAAoJ+pp9YtIWZf7Qp2GStc2A30s4mdlOFq6tx/jvno61GXjTETh8PFdjfgg7PTmpl/RCbWzTrdAgdrM9im+W+AkB7uN24kg6EgB8A4GzSboePgx7EBuaHkwLxYxUU2UtVfUDdV/fV4a9Q3V6tWma32UFk3qnetT6aEI7UAQCoHE8AdM7rZLmoYqPxteYwMy6Wj9/B5qOrxa9ZxyuftlAvXwf5pflogYaYD1Y4ciSpdqQWtlCLba2adfizpDnYHFdm87PzLgwZR8JXNlcm0MPvLtILbDeBOJ0AgMyTTjKsO7JwJIBOn6pZOG3sGDP6zLDWuRhI0ydie8N2BMcxXw3xeiNltgwW1OObou1fdfQoBAB4x47VrJPnzgMADPQMTKiP3AeaKDILmf8sPDLed8oxtfdUzbIsJ/NLuwfKvtz+8X62fA/zVY0rGme5f3Zih19mMun5spOFFTMf99nZczuJkA4bIyacVGo8r/XgmzZNs8xeWBj1e1Qh89/bSOoeZ55Mdk5t4+LnA3+J2DWM1ckYN571Kz/fsN0e3h6GdcyC2Nn5Zs6M9unnuDvHQy0WIBaOnmzue0eO1KzD+z26x+i4ssNO9jLSPqLM8FiTS1iMgs/pM6yrhHd09HE9On09lpBeYLsJxM58Gu5BxgtKSSa7AbJcWQY1o8EfOOW55RZ7Fw1azgKPxL4sSMfm6uSsEjcLanGpLCiRfvjYzeQs1fbheh1s0S50FGrW0YLH7sGw/GGW94s6flYu+3+Z+nk4bA4MzR+qWtY/pz+m9GaLnr2gMOr/SmQ42ctIaVap5f799oTf4rwiFiDi6j/A8v5mUOIrUX2AasHVj/nebIoXLu6rS2rOOZ3wDFW/1q7ycmRMiM5K1y+7HwocBSldYF0u9jXmKVe5P7n16Tg2V5x92Hjyea6HccXjUhbkBBs739hnyITiCZEgJ+8Y4/G0Z7Pnil6QFl8Q1eIwmsC+8vdlG3/B98rshWxXNlx2l2FdDltOTtyC6h058jsR5JTmwXYTqMjMfeEaY1NaY4CZwPyCtrlKDdwkebgjOZMiOXQE6AnY6poAdJqFgU6qgK6JuJ294YqNTZp1OLWgSdSuo4WAGEBVW5Xl/ZQI+pkJLFirfh6CJOBgq7oPfG/zXoREZi/kpk2pPd7UHBRkE7Ffn86hhtKsUuwh7MtE0Ohjsqhpr0FlS6Xp+kJdvGmUB8EkZSIOhxGuUh/PUEUFiC0qQSH2te5DvVAPiUopW2TDQXavBQ+ruHB4isiD7BxFmZ5DVdwCsdjTvCdlPlhw98yh6PlQ0VSBypZKCLUCQnsrDZvhVDvu01UDpxfV+uP9zP5WNuebD8XHHcTiYNtBtIRaEBbDcNqdhvUB5oMN7d8XtS20f3/aB5uGNmiYPZDb164xrLujgQUhWH0w84fdmmrjY8Sif05/9JWjA8nabwEAzo27WLuKBwmVfVUd33yj2ZbYJPuQN6nz7JTY5tcOtNDDqgOrEtqPo72ePaTo1h2adT45+IlmGQ9AC+3fDwAIH9gfV4f7wjfXbY4rA4Ch+UM1TYifVH2CTY1MbKLjK+tBa2bx7ZFvTdf1b2B1xdZO3x1/EK89pB4cZBY8gEoNwV27o35vPMKC51IZyc1fuOwbtscXyl+w/b9hC6x/ExtPMw/8XY27UtNBdJqqYznoypekDhMCJaL8ohTYpnKuMvgL8Nqa+HEl7exlpHCP8fNpUx17BnCfrlm0f/Jp9O/PtHn3xxLSC2w3gfNgc886y7Du5F7Mb2HVtHhcPvOZWjH7cfzxxD/iN8f/BgDgOPt0TOo5CbZTpwMA7DL/DmDmGwDIPu00zbY8w4ez/4/S9puMKGTE/r4ua1xfgJmIfzT0R8YVdZBfykyanhnTVcsdNgcuPu5i1bLTyk6L+C65T537tZQo9rGsVzxxSCxeXvAyLhhygWrZm7vfxAfC5wCA7NNP1ziL5GCVB5s1i/n+HAqfKOfBWmknDk4n8haq84HVeLA3jLkh8WNpILuMmV09UybHlRGnE+0egoMnMzN25gw+5sZ5hVPJg+X846yZs6K2W+XBeseMAQB4jovnuHKMLWLxE4Ny4326bX1Z4uOmwcbn/7NRPzOsE4s0DzaN7zXyPHl4Ys4T6OXr1WXHsNuY706E2GXH6CrYbfao7DldcgzS9VzbNNLQQuT+pPH353g5CUWh13r8xPcd6QW2m8DNrE1vvmlYl5t4q1qt+Rm5uW93026DmvG447M78Ke1fwIANL78EoDO9HXcZwMw/wgAtLz3nmZb/m+ZCS+4XdsExU1ne4PW/VMBMYDntj1nXFEHDVUyV3eVuhlYkAS8sP0F1bJ397wb8QG3rWScRKEunjpxuJ2ZoT8+oG7+vOCdC/DyDnU+r43Y4A6wRbxlaRIcUx1Y5cG2fsjPtS6yjfv9rbQTh3AYjYvVua1qPNiHNjyU+LE00LKbzdX2tfEmUcnvhy9AUfQJc2e0rZL5wIeM/c6p5MFyil/7J9Fz1ioPlrscgrsrNOtsrd8KQN294dzN5n5mhXFswCPfPmJYJxZpHmwaCUOsM/ZbtIQYf5Q/vMyCc9YCojX+LMACd/a3Mj+ieKQOp79+OtqPyAEfYudbLJUDPsSGhrg2OHgyCsmv3Q9+bh1Sh+W+Ap2+v0Qhyj5xR6t2H81wbYUGmZMpxr/p8z42h9SDSbY1bMMRvzqn0WFzRIJr9K710QR/wNFw/Nd7IoFcSqgFiR1NSB3Mr2jviL/nqDy2WfJUEZub2PZwcnPQKiQ5uExs0eaXm4Eo82BpSPv5wjn4ar7TYGsTAKCXLc/wWGo82u8z0gtsN4HzYLNmzzKoyXLdAta1IzkP9tR+p1rrXAykU07AgdYDCE5mPlTOfQU6qQmZ09V9lwAismZ6uU6JLF093jc+oT5O663NmzSDzCJm/hbGqdNDAODkPidrlnEKlWfIYADMVxiLAi/zVU0s1r4OWlDyYLnPryswOG+w6bq+KUxMwp6XG9nG59ysfsbzWg+x3E4lnL2jqSLcddEVPNjACaPiC+UXnY2TGS/dN2kSAHVqVixSmW6UyDSdrLnRucaVfHnPiBGG7Th7sT5xX6wauJzkrL7x47rf1gQAoBOMuanDC1g8RqYz06BmNGKfHd4JiT0njjbSNJ1uAufBRgmva4ArUFglZxPCFi2rIgGxoP16A1gHsYQ9QPjLAQAQmRPr1JDpAjp5ks4S4xeEIoe1ZBoAC3IamJtYggoOry8HYQCe3urn4bA5NLmd/XP6RxYmex57i1cLwOBcX85rtoJfjv8lPjtyHIDfw1mqfa2TQYmvRJPrqwanLLnIA/aAzkWOR6AnBKczIhARC1d5eVzyhF6ZvZAhZKR0gXU4WbCWt59KQJK8wBKZH+6Q+aO2DG983RiMKBxhiQOqC/n+jk1qMrJwJPa37oejqD6SyF8PNh97YXboiLLzZ0m/7PjnVRPtQB8Aez1tKDM4VlFGEcKSeYoOwIKc3EOiA7A8g4cgpGPSPlaQ/oLtJnAebGi/MXesroP5uKymA+RmHZ6TOFGQKmYath+S+X4KUxhP18Ylr9TAfbZmUrodClvnTwbEQNJ+rUAHM2MHqtXPQ5AEzTRve5v3RvILc8kyqTU+7RwfD+6LtYI8Tx5yCHvB0rvWyaCmvSaSc9YMOIdbSdvigVgHWo3ntXbDYYT2qo9nqKICga1bo7ZVNFVgd3B3SoPABJkH66/aF1fG6TF0LztHzkOV/MY89e0N21Pmg+V89NCB6NiMrfVbIz7YwE7j8ZRkE7NoILcHQLXvoXa2f8tB4/Pa17IPOxt3RnjjZiA1NyO4OzqOJLh7d9oHm4YOBLYw+U1wQ/mktppTmFMm1LhrRhiaPxSDc9lXGdnEblLHLvawkYKK3LNB9nANbNPmr1L5waPHs8t0MZPRho4NlvsKAOsOr0toPw5/s/zyUFGpWUdPNIH7pnjAT+hA/ALTGmKLrlbQ2YTiCZqR2p9Xf47P6z8CgLgFJpVQS+auheBOxhnmiUQAQKDsoc/5jolCj78Z3h/NMea5iFO5wHLRAvvmeN4qtz712sReMPjDX+2lKhb7WuIX7ETB+8EFJjhq2juTYwS+1RbY4OAvwHqLcbY7GwCw8oCKDqucizprr/ELNA/UtPqx0LEmmsvf8eWXlvbvLqQX2G4CTzGYM8+Y0zixhPkfrPpvuOB6ic+6SfKuE+7CzRNuBgA45s/GjD4zYD+FaWPaMztN1TZZG5ZrYqrBO46F8bv6l2nWGZw3GNmubJS6rKcR9Ng9uGCwOn/ULPJKygAA3mnq+YYdNgfOHXyuatkVw6+Ax87MpJyTaM/OiavXI4P5orVyGj8590mcPehs1bLPDn6GZSFGrs+aPVvjLJKDVR5shPNb0JmDl/vqTu+fBFfX6USuhk7yUePBDmapN7PGxvv67NnZaPcQNJ/AzOm+E9l4OoqM3Rup5ME6+zIzfKx/0ioP1nc8y+PtKtW+94YXDIfH7kF5TrzpvqE/G/9AP+Pzv2XCLYZ1YpHmwabxvUaeOw//OuVfKMmwvlBbgZ3YVXl2xzrsNnvky63LjkHs30mOcBrfH9iITXWe85cpt91YTeeHhvQC203gfqtGDd1LJT6tYmnC9rfEp9/Tw7LKZQCAhoB1WsfNq27Gbz//LQBEOGjNbzFuozI1HveDNL/ztmZbPK1ZqLJSs86Wui1oDDaiImg9cCEgBvDklict76dEfRUz8wU/XKVaLkgCnt7ytGrZ45sejySa4Nq5ajQTnp93xb4VcWUAMP+N+ZrapnZiR4afmUCb39a+1snAKg+25X02v5S+dU63enWnMf9SE+EwGp5UH8+jxYNt+pa5A5q+jnevCHV18AUo8lczczrXQg6bkJ5MJQ82uJOZrztiuLpWebCtK5nZV8+3v6F2AzqEDmw6Em/6d6xnrh/fgbq4sljc++W9hnVikebBppEwJB2NVA6e5N8q15P7bCmo5X4daj8UEQkQm5tx0ksnob1RfpBKCl+XHPAhtWj7n7iPRykSEAv+YA5I1jm7qYAkB27Z/drBF0bXX6KS7k3PF+F2Qd3/tK9ln6af3W6zg8o+RilJ3mOqIEXGNf7LmnO3E8XR5pTGHb9Dnoeh+H7wecz1YCMvUzrzuyvA4x/UtIetgOs188BLNfBnENdtVqKxgQVR9vOmjoL0fUF6ge0mECfzVWWfrp3Dl2N8MfMD9cq0lqowJf4wANLc6WgINCA4dQyAaFkum48FJ/G8tGpwyuH/vsknGB5rZrZ2O7r79U1sP47snoz6Ik7U5vLNKZujut1lc0W0Xj3DmO/OPTieT8oT+Z9QYnwdYmEndrS72ctMps61ThZcW9UMfNMZ99iR35lggPNgT+tvPK/1kHXaXM2yWI6x2+5Grj03IpWXCniymA9dPGlSfKH8grlpBpszXAuZ5wbWg5oPM2HI6QtzFsyP2qykSHnHG/NFnbLvNcMEt1Qt5/ceOwsQ1NPw5SjOKEaGIyMS1GgWvhNPjP49xfo91B1I82C7CVwsmXMJ9cATFHAepVUkwrtUgpaw4AWpiD1IiaNz2hAXe6A6irU5rsTNfDNceF0PxU5tLp4WPHZPcrxLAG5vJoIA3CpJ+gEW5NQnS51/2jurNwbnDYaN2GCTtTW5xmbUMRwssI0HO1nBFSOuwIB9PQDcDadGH5NFia/EUgCOUw7q4QF7QCcPNpHAus6GnXD1UR9PNR5sD28PlNCSCFczFXA42MuptzS+H1RiLzr2MlbGF1biMfZBDswbCAkpinaWTzc28cagvEFwO9xwFNXDPcB4PDlP3Z5fYFhXjQfbKLIv4J22I5gQVxqNfE8+jss/LvIiZga2nBy4ysqitrnK+iOwNTHlraOJ9BdsN4GnWwtW7DGse6iN+e44zcMsuNl1X3Ny1ACyh1FO7FUy3y+koOnIf4f2aR9DamP9NuOj2hM0vh6xCIgB7GjUlpkzA7+c7s2/v1K1XJAETY5oZXMlllUuQ1gMQ6hlZnQ1U3FHmJnXuLSdFXgdXmRQ9lDS82Ung5r2Gkv0Gs6/VKbA5EFqe5qtj2ME4TCCO9THM1RRAf/69VHbqtqq8FX7Vyml6YQ7mNm3rUJlzLmpfjvz23PfpdRhnOZzQ+2G1OnBys+Q0J7oa72+dn3EB8ul9HSbkfOJC7XxWq+xUJMz5Cbypn3GnNvdTbvxcdXHltK+Ss3NcdS0wJYtaR9sGjqILLDGQT3V7czHYVVDkZstE9HmHFs0NmIuJBUsuMp+QE6QEJVoQl5gNRIDKMGDMtSQ42YmuXXtifFZt9Un9zYb7JB9hge1E11sb1Dn8XIfd4fQARpgvio1Hiz3Y2mJNpxUepLqFwLAOLgf1jNBBTPXOlFYWRi5CLbk71xY+AJb0ZRclp2AjjCEcDg+UQcFTekCKzrZo9GxozKujH+xF29nC1KYv2i0Gy+wPK4hFSDyl2fHhg1R25VBjXoCGxFI8rNot/b9medh1qv39saLeojy/HfvNX6B5nEM/GXTLGK1pP2bjV8cjgWkF9huAk8xmH2qMadxnCwHZdXsdmoZy0HMfbFWcNvE23DdmOvY/nNnYt6AeXBPZ3y/KB+srA2bOWOGZlsZcq5WnpNYDeW55RicNxi9nNYl8Tx2D84aeJbl/ZTILWbmPu8J8fqfADMRn1F+hmrZryf9GgALcsqaw3yHxBFvAuOm/uNLjldt56GZD2H+gPmqZVvqtuCDEMsWlXmydk7kZGCZBztV5n/mx/NgZ/dLgqvrdCLnTPV+HC0ebNZIpqGcf1y8T9pZVIR2D0FoLEtDmDGZjacZF0gqebDugbJm7dBhUdtn9Z2FgbkDzfNgVcYxFsflH4ey7DLVF8DqcpaEQigyTvbP7xUrsOXkIPfCC6O25V14YZoHm8b3AznuHNw37b6oJOJdATuxp84/dRThsDGfdFdyeLkeZxppdBe0eOpXjvhJN/Tmu4H0AttN4KH1Dc8/b1h31YFVAIDKlkpLx3hqy1PWOqXAdR9eh1998isAiHASm157DUC0PBb3gzS9/ppmWy3vMh1Z4bC2j2fTkU3Y1rANOwPmc+FypIIHW7efHTe0TCUVHJgP9vFNj6uW/X7N7yN1Ovmb8dSomjZmQlteuVy1nVNePgVPbX5KtcxBHMiUXZ1Nr7+uWidZWOXBNi/h49pp9uS+tRd3vJh4R8Jh1P/3v6pFR4sH2/gp03it3Rifki9UWQlfgCJrrcyDlfnA4YPGOb9TyYP1f824uv5vo/2iH+7/ELubdpvmwTbJ3G2hsVGzzleHvkJFcwW+rY33wZKPmQvKW2tMzUrzYNM4qqAdxgnCeWJsUbL2hZSM9mJTsCmiWyp1dGDicxPR0Sr7dqhi8eCJz3W0XiNJ0Kk2H5f7ZsK0e/iPPOjMFkqcyyhRKcINjYR4KsDf/rX0eY/4j2hyZO02O4h8+aiJYJqjAT7mPKpWCas+tmMNVNZalcT4+cA5ug75dpRkPios3p/JguvBJsu/NROcxe9Pvxj/vKquY3778pwBSfXj+4j0AttN4DzYnDPV/XpKHN+T+Xh6Z1kjcvNk//PL1f16ZiHNPwUBMYDgSSwIn/tdlX/HalIqwWXqfNNO1KzDcV6+sc9IDXPLtHmTZpBTwnxL0pSxmnW0/KN57jyUZpai0FsIz0jGo+W6sEpwE3si2rV2YkeHzIbR44gmC865NoPMU04BEJ2LmNMvFpQvSKof2Wdo7+8eFi+pN9QzNLU82MxcAIBtdrz2Ln+h2DaHjXHmdDkns4lcxFbkAI3A6XI555wTtV0p3ZihEVOghKsvm/s8J7EebptwW9y23TaWwclMbICDODCqcFREP9ks+FyL/D6p6zSRU4kfFA+WEDIAwB0Aciil5xFCbAB+DyAbwFeUUvVceF3RF5kHa4acnuNhEbYei7k+KSgyHBmRaOJEQXuw/aV8FszAVTyATm1Yh47YNA/ocuQZB0EkqgebFO8SgMuTgQAAV4H6eDhsDk3B+1xPLgbnDYbL7mIBYA5HVCBY5BiyDiiPyLSCM8rPQE5fG4Dbda91MijxlaA007zYAk8wwccX6OTB8qQaCcHp1OT6usrLIwtCpB/EgVJXaUp5sHY7ezRm9FQJupOjlZ29WRkXnFfygbXQN7uvpgUjUTiLo++Zftn9QAiBo6heN4E/h83Nxs9M0JCqHqzMg/02vBeToS+6nuXKwtCCoZG4BTOw5eTE5Qtw9u6dDnICAEKInRCynhDybhJt/I8QUksIiYvNJoTMJYTsIITsJoTcrtcOpXQPpVTpkT8TQCmAMAB17kQXgZskA9uN+Zuc1mE1/VxACKBD6NCkl5gF2cl8RvZK5mOKlqtjZiq9EH+xiZmaY3Ur1bDRbyyvFYuAGEhaHs3fyvxP/j3qUnKCJGBznTo14GDrQSyrXIbmYDPCNdWAIKhq33KJrkQky+w2OxwC+3KK1cZMFWraa/DVYW2ZuFgEZbqQ0sTIzeBJ8ZLDYQQ2q49nqKICHV98EbVNoAJWt61OLQ/WL3M7t6uMuZzJSdjIuJmcA97pHtDG2pq1qdODDTPTcOwzZE3NmggPtuMrbYlFDqGBzf1wtbEPeVXVqrhtRKYnte4xjp9oDDbilZ2vRLSRzUBqbo7jPndsWJ/2wcq4CYAqSZEQUkQIyYrZNlCl6lMA4uxihBA7gIcBnAZgGICLCSHDCCEjCSHvxvxT+zQaAuBzSunNAK61dFbJQl5gzSRf4Lw2f9jYX6sEf9v8ssa6duIJvU7AxGImg0VqWHCS/YgcBKHwS/G8rFx0Wg22jAwAQFBHb5J/1a1pW6NZRw/7W60JIcQiJD9QaZ12oIcWf5VnZjrUfghE/vJRe5ngD5V6v7qw9Wn9T8Og3EGqZVvrt+Ld+jcBAMIh66L0ZmElCQZPqkEViUf4IscDuhKF3ssYT4wAdIqfd0gdKV1gaRazQLh2x/fDls0sOT12s3EUZQ1gMzmBrWo664F/OftjXkaUWqtmONM2L7OMBXdrc5c5xezt3fFCE34Xsxy49pqblxKVIpxws4hNZBOqTJ2ubleiSxdYQkgpgHkA1MMvgZMAvEkIccv1rwbwr9hKlNJPAKhJwkwCsFv+Mg0BeBHAmZTSTZTS+TH/1EJYqwDwJ+pRjVDgZrWsk2cY1h3Vg3Hxin3W0gieM+gc40oa+PnYn+MnI9nHvnvmDFww+AJ4prL8n3zBBDo5sbG5QpXgepm2LO38o/1z+uOk0pPQ06mdclELHrsH8/rPs7yfEjlFcj7WSRNVyx02B+b2V/d9/moii7YWqYjsedr94A8pzmuOxV+m/yXCXY5FVWsVPg2zr0u9a50MrPJgfcczfrPSVMd9sDP6zEi8I04nsk9Tz2Ucy4MlhHQND3YsG6Oisnifqau0FO0eAtsw5oPl+X4dhUeXB8vzXseazK3yYLk/0+bRNnEPzhuMySWTVdOF7hvEvpGk7Iy4slgkyoPNOeusqG25Z52VNhED+DuA/wPUyY2U0lcALAPwEiFkIYArAagrLaujNwBlypwqeZsqCCEFhJB/AxhLCPk1gNcBzCGE/AvAJxr7LCCE/Le5OfGI3O86sl3ZuOuEu5Lzq5mAjdi+s3qwgPUo70SOkUYa3QUtHix/wUwjHl22wBJC5gOopZTqOgEopX8BEADwKIAzKKXW8gFaAKW0nlJ6DaW0nFJ6L6W0g1L6E0rpzymlD2vs8w6l9Kc5OTkp7Qs3J9U/bRxXxfVDrfpu/rDmD9Y7JuPKZVfiFx/9AgBQ99//QKISGl9k3EaliU5sbAIANL30kmZbjc8xjVMui6WGDbUb8NGBj1AdNvYDxSIVPNgj+5ifOrzkA9VyPR7s9SuuB8C+YI/89a+ax+Dm16V7lqqWH7/4ePx3ozr/00EcyJKtao0vJcEx1YFlHuybrG5YYbLmPNhntz6beEcs8GDbw+1dwoNtePcdAMC+bfHuFf/GjfAFKFzfMM8X53mHq4xjDFLJg237aBUAxOVttsqDbXjqKQAK2o8Kvqz5EqurV6vGOohvMx5wRqsxxS7Ng00dpgI4gxBSCWa6PYUQ8lxsJULINAAjALwB4LcWj3EQgNJmUSpv++4gbMxh42+NVn1MSl+MVQSEQGe0Y1jA6GdGwx9QExuQebA6XDwaln10Nu0oz1T6zxKCTL0gKpxOsxCpqPuQ4v5Cgapfqw6hQ1NzNuoL1sScORqIjLnKJbOqXZwouMZu6htm7arNS65/G+El85gEHZ53VyBVmrk8aJHYtJcDPcvS3joW4DgwTz1+IHKc7r7HuwFdtsBSSn9NKS2llJYBuAjASkpplJggIWQsgP+CRfNeAaCAEGLls2sdgEGEkP6EEJd8nHgv/DEIm+yDzT3vXMO6U3sxH6aWXJoW3HY3HDYHzhp0luX+KSGezfyCwZmMJ2fL6oxLs8t/Z8/X9j3yEHu9fMUcF+VflFAftfIEm0Veb0aSp9NV9D9laPm0e/p6YkjeEAwrGAbvGJbD1qPC1eQ+9JP7WM8l7LA5EPIw/2b2guQ4pnqY0muK6bpZp7J5oczBy32wyfj/ASBH474gTie8Y8ZEfvMH/8lZJ6eWB5vNgu68Z6j4guWFYveZrB9ZJ8t84J7G8QNjeoyxRFHRA3Gw88298IKo7SMKRkSkLc3wRV1lMg92ivHY/+nEP8Vt22ljwW5ZBnnV+VidPfBsZLuyDY+lRGz+aT73jnV0d6KJDAAXUEorKKUSgEsBxIWHEUJeAPAFgCGEkCpCyE8AgFIqALgBzI+7DcDLlNItR633yUB+WzTjqPe55IhGu7Wk/RQU/bL6WZ7Mcchl+0vZrB9Rb7oy2d2ek6u9P6+TbdyPAoexJmUsPHYPCjzW91PC4WIBHk6N83DYHJp8Yq/Di7KcMngdXtjcHthzc2FT0Qbli49Vkj0ATC6ZjHv6spSM9hS7KzhKfCWW/Oxc81apD8x5sLnu3MQ74nTCoaFN6uzbN2oh437vImdRSnmwNnmx9uarXA+ZpsP1cLkGMOeE66HYV2z5RdkIsUn6i33FKM0qhaOoSFdgg4P3W/nirIXemfEhLs0Cs2ytbdNXuOFj1Te7r6WYAltOTlwAmaOwMB3kxEEpXUUpjUuDQyldTSndpPgdppQ+plLvYkppCaXUKX8VP6EoW0opHSz7Vf/YdWeRWnDzmhm9Rq7najXE3y/4UdFcocnfNAuymdFrHLtZPFkUD1b2JQe2a8vFcT+tGcrA6rbVlvsXEAP4utaY76eH9iZGtejYqc7fFCQB3xz+RrWszl+HZZXLcKD1AEIHqyA2NSGsIqnG9XwTlXIjskkwsK1rhKZr2mvwefXnpusHdzHTINcDBTq/UrbUJ/GeGw6j4xv18QxVVKBt1aq4461sWZliPVgWL1C/Mb4fPJNTcN03kT4B0Tm6tfBJ1Scp5MHK6QtjpNs+PvBxhAfbvtp4PAWZZhTeb0x1e21XfM5xZztzi3TocOGBThPxM1uesUTTkZqb0bE2WnKzfe3aH7wPNg09yG/BoonoZO5L5TmJzWJ0D2au1FoY9HBK31NwYi9GByHNbGEgLfKDVBEtyxNmSM3aDxeerYo/kNXAKSwbOjZY7iugzS01CzEs+07btP3WjUF1jiy/znua9sCRy0yLaonfub+wLaQe7HXuoHMxLH+YatmBlgN4vY4FrIgtXRfRfsQfnyBDC2ILmxdK/zt/iCbL9xQbtPnIVME35dnNjghHUuvjK2ZfTBkq3E4+n4v3sTkvtsqxCSZyAlvlf+rBWcKyXcWKkSt9/GoJT2LBvw71EpgUZzD3xtsV8R64xiy2jBjpwXILQ2Ow0VKiCQAQ6qPvb/5ScKwjvcB2EzgPNlPmiOphWCF76FqVi1s4dKH1jsm4auRV+NEw5jL3nDQNVwy/IpLX1CYLPQOdPNgMnTymmSdNZ3/Ytc1C/bL74cIhFyLTps2V1YLH7sGp/ZLzyWT3YKavjHHqHFWHzYGZfWeqlt049kYA7MGm5x/lyTRGF41WLV80ZRFO7qvun20KNmGtwJRMfMcb55dNBFZ5sBnj2bVSmqy5GZzHDSQEpxNZs2apFsXyYHM9uV3Dg53A+NA9S+IDd9wD+qPdQ+AZUA4AEZ+wXUdPlSOlPNhRjB/vKIw2Ayt5sDlnGY9nVkyeXzUMyB2AswaeFVloldg3JBcAQN36JnKvw5swDzb79NOjtuXMm5c2Eafx/UCWKws3T7gZhUn6OY3wXdWD5f6kroySTPNg0+hu2IlddY7/Yervu6E33w2kF9huQoQH+/gTBjU79UOt+u6uXn619Y7JuGTJJbjmw2sAAHUPP4zWUCsanmXcxmgeLDPlNS5erNlW3UMyxViHxfDN4W/w/Pbn0SFZlzlLCQ92L/NrCu+oa7Xq8WDPfZtFvIqSiJo779Q8Bk+1qGZmA4CRT4/EwxtU6diwE3snD/a5OLZbSmCVB9v0KvPHhWviebD/2/y/xDtigQdb0VTRJTzY+sVMp3l7xdq4srbPVsMXoKAb2ZxpfkvmAx84EFc3FqnkwTa/8SY7bgz/1ioPtvavfzOs80X1F3ht12uqLgThmVcAAD6qH4RZ21Gb5sGm8f1BKrmIU16YkrAKCOVvvakL8jymwCNnBSpEfOupRiopKN8ndBkPVp6zVC0yOVLWNYc2DX5f6fDLTSHJObuzniVpGZQfL9GoRFdmOjtWkV5guwkRHuzFxrzP6aXMh6kmFaUHj8ODIXlDcP4QK9kn4yFewDiugTmMJ2dT0G049Sbn7LM19+eSWVkz1X2YSvy44McJ9fHcQcZ8Yj3klTJ/GmZq+w4vHHKh6vbSzFKMLRqLWX1nwTt+HONqjhgeV49L6s3uq88XVIPT7gQymL87Vv8zVXDYHJZyCHO/mJIKwqlkWtfKLPIuuVh1uy0nBxmTO33QPIr4kvxLUsyDZf7U3PNV5pW8IO27iM2VbM4HLjGWTDy+5PiEaFqqkOlR+T+KSi+A8cXjI5QyM3xRVznjgJvJcf3gSQ/GbdvpYAFIsX7SWPDgq1sn3IoctzWqWaxudvb85DSujxbSC2x3gfNgVXRDY+FxsEhJqwR1SikKMwojpPOE4WP70wzWjyi+oRy4pBQAiIPNzjRSVbihsci2W+fseuyepLm+DqesWZuhPh4Om0PzwWi32VGUUQSPwwPidMLRs2eURqqyHgB4ndbHo192P9xRehcAg2udBHp4eyDHZf7BZ8tg5xGlDyybKXxO43mtCacTtkz1ax3Lf+RfRdmO7JTyYHlb7ux47V7KebAyZ5rIQX9KPrAWct25KPSmSM9XzhwVy1/NdeeiwFsAR1ER7DnG9wWx2WHLyDB1f6r1vSXMoqg/rtNXwuJjVegtjFh9zMCWkwNbVvR52LOz00FOaWiDUxs6vllvULPT98pl68yiI9yB1QdXY32t8TH0QDYwGoBjm6z/qUgHSP3MMRjYpK3jKjbUA4KAgI5cHcd7ze9FUgqaRUAMYE1NYjJ3HG2NTGypY6s6f1OQBE3Zv/ZwO5ZVLsOWui0I7duH8IEDqjKEXM93R0NiWqlEloXz61zrZFDTXoOPDnxkun5gK/NBRmgq6DTZbqjdkHhHwmG0r1Ufz1BFBVqXLYv85l+wS5uWptQEGW5n53R43WfxhVwi73OmSxuU9VhFHaoaxwf7Pkg9D/abaBreiv0rIjzYVjlfsR6E2lpIHR0I7tljWPff3/47bpu3g415aKc+D5aP1SMbHkFH2HyshdTcjPZPP43a1vbpp2kfbBo6kG9SM/lE+UPLapTqKX1Z+P2mI9bFyE/vfzpm9ZWpEiHWRxLJPdu5AHLSvd55uPozE5QeD5Zrqu4J7klIUSdR/zAH/yqBoH1sHsATizllcwAAm+s2w11WBiA68CdyDJ6LWMNveOmwSyOc2lg0BZrwQt0LrJ1Q1+X51TpHNUTGXOV9KCRZ42zHtR3U2V8x/zh/en9oP6heFJ1F2Pr3BQDk7I/n4zpll0fJQfZy2ZmT2fj+TGWkubucuTUC27Zr1qEmNGpd/RltKKSzwHL3xhc1X8SVHSyQv/YNeLDZrmzYiA37W/dbmmdAvBCBmfM6FpBeYLsJPD0Z19TUw5D8IQDUzTN6uGLEFdY7JuNHw36E8wYzLcmMqVNw3ZjrkDVRzkWsMCXZfMxc6R03XrOtzBnG+VD7ZPXBTeNuAqCfWFwNHrsHM0pnWNonFlkFLP1ehswtjIXD5sC00mmqZT8d+VMAnAernROZ+52GF8b7ZwHgtom34cTe6n4wgQrYILIvxozx2tc6GZT4SjC3TF3zVg3e0exa8ZSJQKcbY1JP43mtCacTmdOnqxbF8mD7ZPXpGh6szDXulR8f9+AZPBjtHoKs3mXst+xvN2OyTCUP1jt2LADAlhnNHVfyYLNPMx7PrFnGsRFlOWW4dvS1ABBnYaocGm9GV0OPjB4JSdvZcnLieNFZs2enTcRp6CCBL1irC4/Vt0QlOsIdkawzPurCtaOvRa6cBCLqBpO//PTOg4aMv2YESYj016qpLyAGkjpXAJDkY0ph9b4KkqCZSYt/rYmSCBrUfrM2+oJtD7drHsNO7JEg7FSpqMSiQ+hI7AtWEYXKvyKT+oINhzXnjOTviPp6ESUx6bFXPU6wc0xjQcNhuEMUElfR4V+wJqJxg0IwZdmcOi0I0QteQAzAL/ghdXToqjtF2jFRR5CEiJUo9jn0t6l/MdVfURItZ6MDWCrO2D5KoSCkDuuUvqON9ALbTeATv+HpZwzrrty/EgBQ2Vxp6RjnvJV4tOlVy6/CL1f9EgBQ/9hjONR+CA2yHqykyLnKObFNL7+s2Vbtg9oaqRwbj2yM+He05Nz08Pz25y3vo0T9PuYflt5fpVnnma3qY3XJkkvYvlRC9f9pv6FH9GD3quvBTn5+Mh7bFJeKGwCjAmXKz2U97d1k0BxsxvuV7xtX5PXfYnze8OHayDb+AH1ua3JcXa5RGguhuiYqF/Hn1Z9rXrNk0CjrNG+qiU8z2vrhh3BIgL+CpRZsXiLrwR40VspcVbUKh9rj3QcJ9fEF5jKITZ352cHPcLDtIKS2NrS8/Y5hO4fvvc+wzrpD6/DkZsY1j11gxccYBz7LqR8dvaluEx78Oj4K2RCCgKZXXona1Pzqa6Ze3Lsb6QX2e4xEfJlamP3qbAQT9XNaDFr6rvHluFlUoILlAC2rx0gjGl2VPYv75NV5sLL1qUuObAFyB/R0XM01k9z9uaWOBQam9WDjkV5guwk2N5NHy7vUmPfJ9UPLcsosHcPj8OC0/qfhkqGXWO6fEuJCls80MI/5xaJ4sHIe2tzztbm2rr59AIcjwhfUw1U9rkpI6uzi49R5k2ZR0I+R5G1zZmjW+fEw9bFy2pyY2nsqLht2GXwTJ8LZp0/EP6kEl/o6rUxFY9QADpsDrkyZm3nBBQa1E0OWKysSsGUGnJvoLO7Mkc15sDyPdaLIv+wy1e2Onj2jNE65teOXxb9MLQ82h13r4h/F94MH9h26nPGZOf+T6x7r4aTSkyznFNcCpwXlXxEdazGl1xT0zuwNW0aGKb6oe+BA2DIyTGnHPjLzkTja3y4XYzfk6GhCA50v/A/PfNjaPW63I/f886I25Zx7DmBCHrC7kV5guwvym7FNhS8ZC6edTSSrDxCJSvA5fcl//bjY8amLtRPFN5TfntV4n50gsGdmmtLLdBO3ZT6jx+6JqKokCpvMUdUaD4fNAbfdrVpGCEGmM5ONk80GW1ZmFDdUWQ/oHE8rcNvduK3X/7F2TMyZRJDpzLR0HSPjqfiC4jxYly2JPjqdIG71a23z+SK8U6Dzq8hj86SWB8u74lXhHMvHtMuc6ch10BGz4PA4PMlxhKP6IfNgY66V1+FFhjMDtsxM2LwmxpMQ2LKyTN2fPqcv7joLkgC/C/jgoD7Fiy+wam3owZaZCeKOPg+b22Mqh0B3I73AdhN4gEL7l+rcSiV2NjD/oFVJtvZwO17d+aomf9MsyDrGu3RuYj4nSRFkQuVAg1gunhJCXR3EpqY4WS01vNH4BpqD1uTYAmIAn1R9YmmfWLTWM79YhwbHVJAEfHZQhRMpY1nlMnxS9QmCeyoQ3LoNoap4fxw/r631xtdBDREe7Hrr8oNmUNNeg+X71HMxq8G/kdG/uGwd0BnAte7QusQ7Eg6jLYb3yBGbi5ibK99ofCOlroVQGzung6tXxBfK5uO2lWxBCWxh42mGl5nKXMRUDsiLfYYoebAty4zHU6g5BOHwYVM89Qe+eiBObjG7A/CGAGmbAQ9WHp8H1j1gmQfbujJ6HFpXrEjzYNPQgfwGRxzGb408A5DVN/SLjmNpGLc3aPPktHDOoHMwr79s8pFNUdQhv6Er+8HPw6n9lewdzbidQR3B9RJfCXpn9kZ1uBotQWPCfiyS/UrnX7DQUa3hUmyxuGgIu87fHP4GGWMYdUKorY2rx8dPyxJx7ehrMbF4oubxn65jgTewd50/1sp1JJH5kFw7qm2bNP9xysv2wPaU8mCdI4YCAIqq4rV73UMYba7PIfYy0ZnByfj+JClMyO2VKWWBbdu0j2fiOnK5PT3B9dKsUvTw9sCmuk0RfWqARcbv6iXzYPcf1j1O78zeyHRmYnP9ZsuR37HnYXZ+dDfSC2w3gd+UGePGGtYtz2WEcp5f1CyuHpm4ms55g8/DgnKmbeqbNAm3TbgNOWMnAIg2SXFznWekOn8UADJPnmF4vJLMEvx87M8BWA/O8jq8OKHXCZb2iUVmHvOLeYerC547bA4cX6KueXvJ0EvgsXsgUUlXD5anc+S85lhcN+Y6TCrR5o/uECtZHzW4usmixFeCU/oYa4NyeIaxa2XPiufBji0ynteacDrhU+QbViKWBzskf0iX8GAzT2DzqVdmvF/Vc9xxaPcQ5Bb1kX+z8TSTlvDUslNTx4OV+dCxL+lKHmzWKer6wkqY4cH2yeoTuT+VwUoSlbBtsLnUn2U5ZZE2rMCWk4PMadG86Mzp09M82DR0IJuZzHC5ggJ727OqHNIYiM9CYxYNgQY0BZoAMPPPpcMvRbbI/GpRkbKizB/1a5+HktajhZAYipjArS6wfsGP1lCrcUUdCLK5TdQ4D0ES4kxjHPX+egTEAAQq6JqtuIksIKhHY9d21GoeAwDs8u3aVfy/en99JJ2jGUgdciYjMZ4Hq/zKsYxwGFKb+nUQ6+shNjdFfgeEABqDic9zLQit7PhqvE2powO+AIUQZvel5GfjSXWygHE0B5stpzzVgtQuX+OYyPXmEDuGUFcXZb7XgpkUjyExhPoAuz+VNDq7zY5HJz9gqr9BMZjQWEnNzZBao89DbG1Nm4jT0IYk+9Man3/BsO7HVR8DAPa17LN0jHlv6Ef16eGGFTfg9s9uB8A4iXua9qDxNaYtGcWDlf9ufv0NzbYO3/dnw+NtrtuM+7+6n7WZAL3otV2vWd5HiYaDzL+MD1dr1nlxx4uq269YxqI4RUlE9a23au5f085SyWn5OWe+MhNPb31ac/+sADPFNb/+umadZBCSQpZyEbcsZXxepTmcL0gv7UiOq9v4vDqvWWxqQscXnXmK39nzDhZv09YiThRNjzHt34118T755nffZXWqmcujdTnLjRyuqTZsd03NGssxBlpoeEqeKzH0l3WH1rFFXJKi8jZr4dDvjQXTvz78Nf7xzT8AxNNtwo88BQDIceXqtvHRgY9UcxmbAdfc5Wh5x5jfeywgvcD+AJCK6Moz3zoz8Yw533MeLEcqecdqIBYUSH4o6LK5ojdn5SKp2/VgI0TY1LRjEspr3hHuwHY5aG9A7gDT+/1QkL5juwncj5n/kysN657aj/FHB+ToT2AluBn3utHXafI3zUK8nGli+s9i/rkoHqzsB8m7RJtr6yrrB2evXoZ6kQBwfdH1GFow1FL/XDYXLhumzps0i8J+xwEA7PPVtVrtxI4rR2iP1ex+s3Hn5DuRMWkSvOPHwzt2TFyd0kyWJH7BAG0/rR6ys1jC9byFCxPa3wjFGcU4o1w7l3Iscs5lmcKcPYsj2zgPNpk82LDbUXD1VapFrv79ozRO+UvNfaX3pTQZhzsnD0EH0PeKa+IL5S+4xmvZ+fP8064+fQzbnd1vNsqyy1LSRx5YWPDT6FiLk/ucjIG5A2HvURgZIz24Bw+Gq39/ZJrQa35q7lNRC2lYCqPS04ZwXiZyzjpTd1/+5bv07KXI85jLXwwwOb7Y50vuxRfBlmNNU7Y7kF5guws8+tbM1yXhu5h/ZeYPHiu6i9rH5x1Q6UekTLtvlFLAbk8pT1EJG7ElfZ7EYDwIIbrHiJRRCmKzqbeTwDgq8dPin/EGEtrfCDZisxTlSlTGnu+fVLSszab9VWazRfFu+VdRSua5EhSQbOpZkniWpwjXmZ+qiXGxEY25kUgXKQVU5ho/BiE2c1meKAXs5voVO648LqRFaDNMs8nr2nUi9VURM+YAs+Z01fMklUgvsN0EzoNtW63t8+PgvMkj/iOm2+cL7EMbHsLnBz9PoIedIJ8z3qVzA9O9lPydycqldhZw06HD5xVqjyB84AD8m4xl815seNFyzuWAGMCH+z+0tE8sWuqYf7Rjvbp2riAJkZzQalhWuQyv73odwV270LFuHUIqlAcedLbxSGJ6rkTOX93x5dqE9jdCTXsN3tlj3rfFtYx5PmqAfdEALEdwwgiH0bpChX8KFR6sPM9fqH8htXqwbS3whoCqVe/FF8qZnJrfY2WBjWw8xQbj4KWU8mBDIUCS4jjDSh5ss4lcxOHqaoR2V+jSfTju+eIeVLVWRX6LVEROO1DQCtg2m9ODveeLeywFwUnNzWh5L3ocWpYuTQc5paEDzonMMg7tz3CwbDJaPEw12IgNV41kZrbdTbstd++SoZfg3EHMNIxsWUXHJ4fjKzP32NnftqxoySwluPSY2qLD0TuzN6aXTke9UG/pRYIjkfSKSjgccuahDG3KgdYxrhjOzKFra9Yi8xRmRhfq4pOCcBNmhlMlOxCAWyfciqm9pmoef3EDC4izmZgzicIKFcyWKWcyUmQw4l84nJKUKPQoGMosT+OLx8Pr8GJ9x/qU8mBdJzC6VMnheOUiTq0bcESe+zyjkAl+cmyawWTgmzIFQLwerJJnbS8wHk+eIlFPrKBfdj9MLpmMPc17ItHEADP7ri+Xs3dV1+keZ0ThCBRlFOHz6s8tq+o48qNNyvZ8a5TF7kJ6ge0mcB6sd9RIw7o8B7GVRcRpc+r6DI0wf8B8zO7H/JGZ48Zj0QmLkDeK8e6ieLCyNqxnqDp/FDCnB1vsK470NxEe7Pji5DRSfblMa9c7RJ2j6rA5MK54nGrZ2YPORll2GePB6viZs1yMLzowd6Bq+WXDL8OYojGa+x8Ee/HwDLXmozaLEl+J7gIfC89gOX+zQo+Uv0SMKByReEecTk3NW1d5OTJP7uR2juoxCj8Z8ZPEj6WBTJmH29PXM67MM2wY2j0EBbmMI+seyMZTqYurheml01PGg82YMEF1O/fBOoqKkDnVeDyzZhpzn3tl9or41ZVRxHZix/5h5ha74/KPS2isbDk5yDg+mhftmzw5zYNNQweyH0dpXtMCTytm5a1PlEQcaD2QWN8A1LTVRGS13G0hnDv4XPhk+iZV6l7KPNhYySwl1LIaxSIgBCL9tWrq8wv+pLmF4TA7ubDGeQiSoMkrPth2EA2BBohURLha+yuAm0+1uK6VzZW63GU7ZV8Ketc6GdS016AuoP8VooQoc0Wp0MmL5A/f5lASfQyHITaqX4fwwYNR86k52ByRAUwlwnXsOqhxlsWmJvgCFKEQc5WIMmfXjE7vkY4jqG4zpvOYgZZJutZfi+q2agi1tRCOGI9n+LB+BiaAPYO4aVjJx++R0QMPDr/TVH9bQi2oaqsyrhgDqbkZYkO0RUhoaEibiNPQBufBNr3yqmFdngPXyoLZGGzEhe9emFjnANzy8S1Y9MUi1tbixdh0ZBMa33oTAKJI35wH2/LOu5ptHf7jHw2Pt7V+K+5afRdrMwG6y9sVb1veR4nGg3sAALaPtX3JWlzbaz+8Fi2hFgiSgEN3/1Zz/8Pt7EG28oC6L3fBmwt0dW19MkvKjMZnolh90DgmgINzLJUPcf4S8drO5HjJsfqfHDQQiMp7/fSWp/HGbm0OdsLHf+Q/AIDN9Zvjy2TO96EGxktvk/3F4UPGOq/f1H6TMoH4epmrG4uNRzaiQ2Av5UrtXC0c+u0iwzrfHvkWv1/D+LKx92f4n6wfRu6F13e+jme3Pmt4LDW0LI32wSr98Mcy0gtsd8Nh7LfhEZKWoogVX4GcOpEoqN2OS5ZegiC039AjeWnV9pcpQ6QLc+gmC84xpbbEIxOVPnKi0o5RLmIj2PjtamLOHBXERtEqcLT0a/nD3mtLnW8TQKfmq5pCjvyVHuJFCeYKTxZU7gdJ1XwwGd2rjCTe17IPu5sqAABl+eW6+/EMUC6bK6U5mY9lpBfYboJN9l0WXG7M35zVbxYAazxYbqr73ZTfRZL+Jwrp0rMBAIEzZgDo1IAFAHseCz7IvVD7GK5+ZXAfdxyy5xprjV5fdD2ml043rKeEx+6JBBolisJ+zPfqnKfOg3XYHJGgMTXMKZuDB2c8CM+wYcicMSOSQF0Jrgc7b0BiGbZycxjPMu+i5MZTCyW+EpxZrs9ljOrP2WcBAJw9O/2UXNLv0mGXJt4RpxMFP/2palFsLmKJSvA6vPhLn7+klgebm4+GTKDs8p/FlXEXSeh6xs3MlnVQnSZ4sHPK5qTMB0ucTtiyslBwZfTcn9l3JgblDYKjqChOR1UN7kGD4Bk9ylTe4qfnPh2Vk9sv+FHlCyBcWowcnTzcQOczac3CNcj15Boei8OWk4O8H0XrC+ddcknaB5tG94G/LaacH5gIJAlI4svwuwTOTewKnJ+vLWr/Q4UgCV0zxyUJVGvKSjyDUjfPaSlFc02SEs4SJlIRNgoc9h/GOxX6rgtuVUvUgvNdxDHw9P1hgso+2FYTPpJNRxh/lPvwzIC/Ld65+k58fOBj6x1UgPslXeu2sLYVyeZ5wvF2HT5v+PAhBLdui/Am9fB03dP45rA1vdOAGMCSvUss7ROLlloWfNGxTl3HVJAELKvUzuu6rHIZHv32UQS3bUPbypWq0nw8EMvq+XGQAPPd6V3rZFDTXoO3Kt4yriiD65Aqg024D5bnz04I4XAc75EjlgcrUQnt4XY8ceSJ1PJgW1tQ0Aoc+EDFty/fWw1vspzQ/q/ZeAp1xgFFKeXBBoOyVmq0T1/JgzUT4xGqqoJ/wwb4N8X7m2Pxq09/FXkeAWzRzGmnKK0H3Jv06YDcnH/rx7fqilrEQmpujstF3Pzmm+kgpzR0IPt2nMXxNIBY8LRiHofHdPO57lzcNO4mANZFAgDgJyN+goXHsZR8tCejsEiFMhdN4Uvl/h9HUZFmWzlnsFRyesnQ+2b3xRUjrkCb1JZQVChPQ5gonF6ZapKfq1lHTboMYDJzALCmek3EJCc2NsXV4+ZTrWCQu0+4GzNKZ2ge/9UW9kDXu9bJgpuxzcBR2AMAQFydPn7+Ndkzw3he68HZW7sfyhR5p5adil6+XtjQsSGlPFj3HOaW6dMY/7XlmzYNADC0gfl97YUFrF8K+poWctypS++XfRozlQd3V0Rt57x5AHCVlRm2w039wmHtIK3+Of2xcOhCHGo/hOr2zvtYpCI+G87G3HlYXymHU5Q+2PdB5EXMLGLTUDr79bO0f3chvcB2Ezg53zP0OMO6fbLY5LJC3s9x5+DCIYlHEc/sNxPTStmDJHvkWNw//X4UDGcEe5u784HKSf/uQYM02+KJJvRQ6C2M9NeqLJ/X4cWoHslppGZksZcHb7k6R9Vhc2BkoTpneW7ZXEwumQyRisiaM1e1DgD4nCwhAec1x+L8wedjeOFwzf3rwKK39a51MijxlWBCsTq3Ug3uAcyXaMvofKBz85+W5q0ZEKcT3pHq19pVXg7fCZ3av+OLx+OcQcb5dq0i83iWaKJHRo+4Mu/w4YwH62MvOm55EVPygbUwuWRyyvVgYzGl15QIDzZjovF4Zs6YYVinp68nLhhyAYDoAEqf04e6keZebkf1GIWLhliPH7Dl5MA7LpqDnjF2bNoHm4YO5EAJMzw1Lm9lJbzfL/ixvWG7cUUN7GnaE/nydTS0YG7/ufC0suNTUWGKkzmQsTw1JUKVxl/QHeEO7GhgqRgT0YO1Yj5XQzjIOI0hjfMQJAG1Hep83t2Nu3Go/RBESURwt3a6uLDI3tq5zm4sttRv0TwGADhkp6Cgc62TQU17TdTXiREEmauq5H9y10Sd3zyfNhY0HIZQqz6eoYoKhA900tVq2mqwp3lPwsfSQlA+hlpKv/DhWvgCFB1yGTdVcrePHqpaq1JmIg4fVB+rA60HsKd5D4TaWs06SoRU3BmxaAu1YWfDTgDR9+fgvMH4fb/rTPX3cPvhhLLKSc3NCNfURG0L19SkTcRpaIPzYGN9C2pYU8P0L62QtPc07cGVyxLP5HTn6jtx75f3AmCcxLU1a9G0hHFdlWLYosyJbXlPm5d26He/Mzze9obtuOkjZtKO1Zs0g/cq1X12ZtFYwx4y9tXa/lEtru3NH9+MypZKiFRErY727eEOtmh8cvAT1fKL3r0Ir+xU538CiCT6aF2a3LnqYd0hdR+0GtpWMN+fMi0kN/0ly0tufkt7/8CWLZG/H9rwEJbuXZrUsdTQ8hDTLd3WEJ+ft+klpgvMeeltHzN/s5mEDVvqtxjWMYu6fz+qun1H447IPdT+uXFO6Jo77zKss6luE2775DYAKjzYvzLOcKE3/mtfiSe3PKk7v/XQFpObuu0j87rF3Yn0AtvNIF5j/h7nsVqJvuNRxECnaTJRUI8bVy2/CkG7WvSkrJ7i0fEPyxHEZnxUgLWcy6kCka+t5Ew8wlHpX+NSYkrw8eO+WKvgyii61/oogrsH1Di/VuIFkgF3J+TZzcufmYI81ak7nkPOaTp+uYi45OugxpntSsh9tJl4huhC7rfSl64H5f25tmYtDsgv/v16qLtXOPhY9fT1PDbYDUcBP4yzPAbBebD5C7V1VDlm9JkBAJZ8N9xP8p/Z/8G5g8+13kEFpIsZvy0wj/lk7Uo92LxcAEDuudrHcPXpC++E8ciaNcvwWNcXXY+zB51tqX+p4cGyvLquueqamGZ4sI+d+hicffsie8ECVR9iSWZJpG4iKMhmgR255xlzGxOBVR5szoL5AABHcaceLH95uOQ443mtCQs8WJGKKMsuw+9Kf5dSHqwzOwdVBUDZj1TGnAKCDXBcw7i+WTK/Wy8wiyPVPFhHSQnyfxzNEbXMgy0vh2/KFFOxEk/PfTqKx90R7sChTAHS0HL45pyqsycbqwJPAT447wNLwV5pHmwaxxy4GcdBjoGsP0nw7L5zkCTVL7pUYFaO8QvKDw2iJHZN1ihKdXiwOhzZowlJSkn2KEqlhPm0EpVAKLC7uQLv7tFOlwqwsbKsBfsdxw/kqXfsgQdEtHzwgWHdDbUbALAgFLPgC+wNK2/Ain3q2ppmYfuQ8S5dXzDdS859BTr9sdwPpYZwTQ061q3T5Jgq8d8j/zUUbo5FQAzgzd1vWtonFk2HmZRexxfqWquCJOgS6ZdVLsNvP/8twlVVaH7rbQQrKuLq1PuZr/LLGu18x3ogAeaENZNfNhFY5sHK/j1BkXQ+JLF5/eG+JPR5w2HN2AQ1PdjdTbvxyOFHUsqDFVqa0acO2L9UJacyleAUgfoXmS+2Yy0bT+GIscxiavVggwhXV6Pl/Wh+tlUebHjffrR/9hn8GzYY1r1h5Q1Ysb/zeSJQAfltFIOrAe8GYx5sbUctrvnwGrSGWnXrKiE1N8flpm58+eV0kFMaOpD9Hu7+xukPOacw02lMA+AozynH7ZNuh1/wJ8QrvX7M9bhyOAuSogMYTUgslU2BTkXOXflvlw4vLf8ylg4yrKOq0y+7H/4w9Q8I0zD2t2jrxmphcP5gy/so4fYxszfppc0xHZSnTo+5Zfwt6JPVB2uq1yDvkosBAGJzS1w9rgWqxae9b9p9mNVX+yv13Tb2INW71snCivnSWcrmhdIHyP3MA3LNp/VUg3uw9ng6enQG0/xo6I8wsnAktgW2pZQH672QUX/KOuK1e7kk4chWZubkpmElXUkLRd7UcZhzL2C0tlid5Tx3pz9a7zpy5P+E3ed6iTIG5Q3CXZPvQmuoNVpwXRKxZKLMg63XV1A6b/B5mNhzIlYfXG2Zihcr0egZpi2PeSwhvcB2E3hAhHugfoJsoNN3Z2WB7ZHRA2eUn5FY5wBM7T0Vk0oYFzBn6Cg8PPNh9BgyGgBgc8XzYPUI7b6pUwyPV+AtwIJy5uu1+iXidXgxND85jdSMTDmZR78y1XKHzYHj8tU5yyf1OQkTiidAoAIyT1H34QKdQuulWeq8wXkD5unyR9ts7AvW1T81PrxYlPhKMKrQPJ/Y1VdngbWQNzsWxOmE5zj1a+0qL4/if04qmYSTSo31hq3CJ2utFngK4sq4Hmy+l5W5+rDxNLPAji0emzoe7NgxqtvHF4+P8GC9o43HM/PEEw3rFHoLI/encnEs9hUjMNocL3ts0VjdF0gt2HJy4BkRrS/sHTEi7YNNQxtUZJGI4Wpjsy9PsecX/Kbbbww0Yn2tcWpCLWyp3xLhpdoO1WF66XS4GphZR6n/yTmQgg5FIbBlq+HxWkOt+Prw1wAS48Emo30LAEE/M3UHNc5DkISoN3clNh3ZhMqWSkhU0jWzBQXGI9biiK47tE7zGADgFNntqpdxJxnUtNdgb4t58yU3idJgJz+b00OsuDNiQcNhhKrUxzNUURHFNd7ZuBO7mrS5x4nCv4PNfTVTZmjfPvgCFK0hZqXgX35SIF47NhYVTRUpMxEHd6mbZHc37cbupt0QamtVU3bGwr/ZOEVic7AZ6w+z54mSRje+eDx+nWcuecSepj0JcfOl5maE9kVz6UP79qVNxGlog4ZlH+xSYw4f5yZaSQLw9eGvcf2K6xPrHIA/rvkj/vbN3wCwvJ8r969E03JmolTzwbZ+qO1zO7RokeHxdjXuivB2E9GDVfqFEkGz7IN1fLlRs46Wb/iO1Xdgfe16iJKIuoce0tz/iJ8tSJ9Xq3MTr1x2pa4P1B1i16X1gyT8mwbYeET7/GPR9jHj8woNnSnyOA/2vb3JcXVbdXjVIUVqwD9/+WfdHNEJH/+fjGO6o3FHXFnDc4sBICKcznNDm/HBJpJoQQt1Dz+sur2ypTLyt/+rrw3bOXTX3YZ1ttZvxc8+ZMpCSgogAIQeZNeqKKM4bj8l/rn+nwlr97Z/+mn07y7Kx51qpBfYboYt2zj9IffdWeGHKm+CZPOf0iwfbvroJoQ8cgSgMuJQjmK0ZWVpNyCbw20+c3zcHFfq8rWahd3Brq3o0b7GRtGqyjy+RIU/yff3ORLjJdtk86vutT6K4OOppgWc5UqyjyY1TiPcSmdyuY/jweY19amYfWUebEuGPPf5dTjaeseyZq3dxDNEF5xfbZJPq0zZ+sauN3AkwL7g+/TU9/dy1095TnmaB5tG14LzYPMuMJYg4zmB+2WbD27hk/nts97GmQPNcxvVIJ13GgAgcCrLAWtXPOC5H4Qn9FeDs7Q3MmfMQJaJnKfXF12Py0dcbql/qeDB5pcykrx7trompsPmwOXDtfs1p2wOXpj/Aux5eci75GJ4h8fnFO7pY4vAzH7aflo9FGaXAQByzkxuPLVglQfLk80rg444D/b8wUlI6zmdKLhSPQuZGg/2hJITcEevO1LLg83Kxq4SoOyiy+MLqYRWL4H3qh8DADJnsvF09lYPXlMitTxYB9yDBiHvouic41Z5sK7+/ZE1Zw4yp041rPv03KexcOjCyO+AGEBdpgQyaSzcp+jzaAUqYETBCLx51ptpHmwa321wP8kxwYOl6DKN1GMOkgR00dv5xMyJXdLudxmiJMLWFXNLhwdL9TiyRxEp0x6mNGG9ZlESQSiw/sgGvL9Xn17XZWN1DOOHdbbHEHhgSLMJH+xXh74CAEt0G246u3zZ5YYT3wi29xnH1f0pC3IQ2xQ+WDkXsZ4PNnzwINpWrkT7mjWGx3rsyGN4ZsszlvoXEAN4eefLlvaJRVMNC6Lwf6buHxUkAa/vel1z/2WVy3DVsqsgNjejcfFiBHbujKtzpEPfB2sEm58Fueld62RglQcb8cHWx/Ngk/LBhsNxvMdI+yo82NUHV+Nvh/5mmfqh24WWZgyuBva//VJ8oUSR3UHR8OxzAID21TIfWIeGxpFaPdgQgjt2oPmd6AQPlvVgKyvR+t776Pja2F973Yrr8ML2FyK/RSoivxUYs5ci4+v4Oa+ESEVsPLIRl753KVpC8TQ2LUjNzWh87rmobY3PP/+dCHI6Bj5vfqCQfUze4SMMKgJl3DRowaxyfMnx+P3U3+Ou1XfpKrRo4ZYJt8gmt49Bhw0C8CWEgX3g3LhTVa5Oi1YBAEU334xDixZBqNdWgRmQMwD/OuVfuHHljVFBGmYxrmiccSUdeLLzEAJA+vfRrDOmxxjV7Xcefyee3PIkPq/+HPmXXYaGp5+G1BavwpLpYjQrLQrLv075l64b4EP/p7gI+tc6WWhRkdTgHjQIrR98AJvCT8ktJiMKjee1HmLlyZRwlnbSnG6dcCue2vIUPj34qWb9RJD5syvQcuOvUB6Oz3Gcd8nFaHrpJYwOynJ15YxqZ0aurk9WH9MR7+FwGFVVVQhoRCdL118H8aILcSArC9XbOkUJ/jXiXxAlEeGHGeVp27ZtyMnJwbZt8cIFACA+/RSklhbstdlg06iTTbPxzLhn0BBogC/gi7Q1TBiG0D8eQtgPFGZlaB4DAG7sfSPaitoQEAI4sPuAaT9s+OGHQFyuqLaFx/4LGgrpHs8s9K6NEh6PB6WlpXA6zcfCpBfYbgLnwbr69TWsyzUprQTH9MrshZl9Z+Ku1cZKGWoYX8y4htsA5A4ajifnXIti8Ru0YEUkuQTQmSBc+dCLRcYkY9NmricXM/rMQI49x3IUsdfhRXmuMZ9Ytw1fDkIA3L3Vz8Nhc2gmT5hUMglfHf4Knx38DL5pJ6Lh6ac1+wl0+mJjwXNOa0GQn0d61zoZlPhKMCTPvI6rsxfjZ9sU4gP8ock1jBMBcTrhHqB+rV3l5VF6uJNKJmF97fqUL7AZY8agBUCuOzeuzDNkCNo9BDkeVuYsYeNpJun+sIJhpn3FVVVVyMrKQllZmWpKRCpJCGzdCkdxMZwKP3hmSyaCUhClhwTYsrLg6t0bra2tyNIIjpP8fgQrKuDq29cwYGp7w3bkuHIi3PzWUCsaW2vRo8aPcHEesnvo52Ou99fjUPshDMkfYvo6BADYc3Ij8w0AwtXVEJub4xJQJAK9a8NBKUV9fT2qqqrQ3wIPPb3AdhO4pmpon3HWIm5abBfiv4q0cLDtoCXKRSy+Pvw1HDYHXACw/yAm9JyAQ4eWA4jW/+QpH8NV2vzNjrXq6QeVaAo0YcORDeiQOiyb+vyCHxVN8akJLbXRLmvuHlQ/D0ESsKdJXXf0y5ovI5zhVp2UkZzHfKhdnce66sAq9MvupxkE45DphyGda50MatprVGkpWuAcbiX/k/v+k+El03AYwT3q1zpUURHFu113aJ2lPptFx3rmDmkKNsWVBbZtgy9A0Szr+oZr2HhKfmOe+tb6raavTSAQ0Fxc9Y7nF/wIS2FQgYKa4OYqaXdaECQBfsEPSZKiMmZlubLgQgASjM+9I9yBgGDcn1hQUYQUjN5PCgSidam7GIQQFBQU4IgJKpYSaR9sN4EvUq0mchF/U8s0SrUezGpYuX8l/u+T/0uscwAe/OpBPPot47e1LF2KdyreQfNKxjWVOjoi9fjNqZeL+NA9xnqwe5r34Ocrf44QDSXEg032C6b1CPNvO77RNhWtPLBSdfsf1v4hUtb07HOqdQBFLuJD6rmIf77y57rapu4gW7y6KhcxAEuJAHguYrGxKbKN82A/2Gc8r/UQq/+phPJl7u7Vdyd9LDW0/v0RAOq81YanWYzAoQ52P3Z8yV4g9VINclh98dBL5i8cVnf98DEAzC364UPGz5WAEMD+lv2g8n9KSIfZomNEI6xpr1F9YTGD2JcA5TPoaCERYYX0F2w3w14Qn4otFpx3ZkVHlH9JDMgZgEJvYWKdk0Hzc/Cbz36DxVl94QQivFagU6PUnqejx+lwAIIAe7axD9lN3OjlM6Y7xB0iyWhpu4NdWyFTW8c0w6GfCm9YwTAAzGpgy4g3F/IHkJJHaK2T7FrbZInA7oY9h42nmvZtvic/qbaJibSDQOc87+dKbX5mQmxsGclVGSv5mPVZMg82m1+Ho61jzBY64nBgX307Hvt0D95cX432oACPi+DMfj5cMSwbpo3+JiJ8bcQWtZDWdtTCRyXYALic5jSAvQ4vCI6BMOyjgPQXbDeB+61yzz7LsO7kXpMBWPNrcTPrS/NfwukDTrfeQQWkM2cDAIKnMF+qXRHMYZMfsjwBuhqcvXohe8ECZE4zznl6VY+rcOO4Gy31z2P34MfDfmxpn1jklzKfn+dkdS6fw+bAJUO1NU7nlM3BS/NfYhzOq69WDUQq9rFMN0a+Vi0UZDN/fdbpcw1qJgarPNis2SyvrKOw8wWOvwSeNfCsxDvidCI/hvfIEcuDFaiAsweejVtLbk0pD9aemYlvywj6nRvfDypR1ObZkH2FzIOdwXIhO0tK4urGIpU8WBACW0YGPq0NY+7fP8WLXx5AW1AABeAPUby6uw1nLanBRzv0gxyJwwl7Xl7Ufd15CIJbbrkl8vuNx9/AI395JPKbUgrBDpCsTCAzPkZk7ty5yM3Nxfz580FBke3KxoDcAZZk64jdDkd+9IeIIz8/TuBeeaxjBekF9nsKbmY9JvQXu1Aj9ZhDqriJKjjOwxZt+7HAbT5GIFGpa+a4Htf1GOHBAsD+lhCue+4b+MMiBCnadCtQwC9QXPfcNzjQqGcq1lYhcrvdeP3111GnYf6moCAA2sJtqnmbb7vtNjz77LOsLqXoyg9X5bGOFaQX2G5ChAeroXupxJpqxh/VSwQfC57J6YJ3LtDVMTUD2zvMH+Zexfi4opx/GACkFsZna3lfm/fINVLbPv3M8FjP1D2D+768z1L/AmIAi7cttrRPLBqrGTfRv0rdlytIAl7arsKJlPHBvg9wwTsXAIKA+v/8B4Ht8b5MTpf6pOqThPrIebAt7yWX51cLVnmwrSuY31mo66RfBUU2r5Oac+EwGp9/XrUojgcriXhz15u4t/relPJgheZmjNlLsf91lX5IEoobJLQ8xR7m7Z/IfGATvsxU8mAhSXji68MIGwT7hEUJz6zVfnZQQYDY2BjFb+dwOBz46U9/iof/yfIeNwWb0BbqvP8pKOwi4AsAtvb4RXzmzJlREbotwRYmRKAyVn6/H2PGjMGYMWNgt9sjfwcDAQgN0RQ/oaEhLsgp9ljHAtKvwt0FOfdtxsRJhlW5hJkVv9b88vkozy3HLR/fgsZAo/EOMbhj8h1wEAckfAw6YRSA1QgPL4dzS0UULYPnL80Yq81bLPnjH1Fzxx0QW7T1IgflDcL/5vwPv1rxK+xr2adZTws8nWSi8OUVIQCADNWW3praWz2V3B+n/hFL9i7B4m2L4bliIQJPLobUEf+w4TmWma82Hv+b8z+U+LTNjGtCGzAfAEZ1HQ92VA/zcnXeUSPR9tFHsGd1mha5f25CzwlJ9cOn405wKSQe/zrjr3h116tYsmdJUseLRfbtv0TzdbdgMOIT2Bdedy1ali7FSImNlWf4cLQsfS/iLtHDwNyBCSf8v/A/X0RvkCSsP9AGQVKvzyFIFO9sqsXehuj9X/oZS33qLClBuKYG0JCJvP766zFq1Cj88tZfgoBErGOLFy/Gn/78J1BRhEME4LDD5nBi4MCBePXV+AQXvTN7oynYpPk88nq92CCrUWVmZkb+9m/ejJc+/BB/V7gNaCgEKkkYPGKE6rGOFaQX2G4CkQNWlNwuLfCFlfMozaBPVh9VDp9ZDC9guXS3AcjrPwQvz78JuW+vRhOWgygSsfPADkextpKGlm6lElmuLEzsORGZ9swoOSwz8Dq8SfEuAcDtzUQAgFvjPBw2h6aO68geIyMPTfv4UcCT6l/TbjmQSivobGJPA76w7HMiRckFrWmhxFeC/tkWOH4y95InGwE6ebB6LwpGIE4nXKXq4+kqL4d7YOdL0ISeEyIyh6mEd8QINENdtMA9cCDaPQRZcrAa90ErXzy1UJ5bnlCUvCpsNoRM3iodIe1jGunYZmdn49JLL8UTjz4RlRxi4cKFmH/efLR3NCHvYKshDzbDmWFJcpOD2O1YeMGFuPwXv4hsSyUPtiuRXmC7Cdy8EdxtzN+saWN8w7Zwm0HNTmxv2I5t9YlnOVl9cDWcNieyAEi792JowVAcOsDS10mhELjXi5u6Q5WVmm2ZkVer99djTc0atIltlgXX/YIf2xqSy+jS0cbeqgP7KlXLBUnQpLB8fOBjfHvkWxBKEXxfm17SEWbUAi1T/5I9SzAwd6Cm6LpDYL4yYa/1L3wzqGmvwcY689zp0H5GOVFSQfjisadZncdqBjQcVjWxA8xEzN0SADO57mzUT9GXCNo+Ya6C+kB89rGOr76CL0DRINOuQgfYeJqhjqw/vB61fuuZ1YDOL04OsaUFo+7/DO1hbR8qh89tj9tf2Y4Rrr/xekwcPxFnXXxWZJFdvHgx7r//flBBYBrRBl+wLcEW0zxYJSWGiiIWv/gi/v5MZwKX9BdsGrrgPFgznEb+0DvcoS1qHot3Kt7BM1ut5fRV4uENDyPbnY2bwLi6L21/CWM+YVxXqnig8oeKnj7jkb/9zfB4+1r24fZPbwcA9KXG2a1isbbGOJmFHtrqauAG4NyoLd792UF1H/KDXz+Ivc17YZcA4d3lmvs3BFjOXs5rjsXtn96Oa0Zfo7nAOsNs8Qqv/UrzGMnCin+w40vG5xWbOk3/3Le26sCqpPoRq/+phFJ39fZPb0+p7zVy/H/+G4D69eCZuri+r/8b9gWtlwqUI9HFVQ1CbS0WlPnwaoW+mdhhI5g/oki7HRPJE3zZPsw6YxZeee4VXPhjpt6zcOFCXHzJxQhu2QoAICXF8BT00Gyjur068vJ8x2/uwOTjJ+Pss89WrZuZmYlAIACPbBW48NTZuPzmX0bKuUi8d0RyKTm7Gukgp24Cf0Nz9DTWscxzM46pFROxSEU4iAPHlxwfSWuWKGhxIf6w9g8I5slh+ArdSx4q7yjSvoE5HPnGPuRezl6W0vVxWLk2anC6mZksnKNtLtMzuee58zCxuNPvqKZ9yyksiXJEiRwtSwqT45imCvZCRp3g6TKBznnNKUkJt23CnwmwIKfemb0x2DM4tdxKHgmucq2pHK17OJcdzy5TSGwKU/nRwhXDsuG06z/GnXYbLj3eRHpNu3409mXXXoamhqYozjkXnQcApyveRD5t2jScf/75WLFiBWaMmIE1q9bA5/Rh86bN6Knz7Lv55psxZswYfPWV+ZdJ5bFKS0uxbNky0/t2FdJfsN0E7rfKmT/PsO7Ekol4eefLUYLeRhAlEZmuTDx+6uMJ95FDOn0GgDcRnD4Ozk27YFfw3bhgfNbs2Zr72/PykH3aXPhOUDdRKXF23tm45vhrLPXPY/fgoiEXWdonFrkl/dACwDtdPbjGYXPgvMHa2pqTSibhL8f/ATswFj1uuRmeIfHi0zyn9Im9jfnAqn3MYuOfMeuUhPY3QomvBJN6GgfdcWTNmIG2D1fAUdC5CLlsbLGd1994XmvC6UTuhReqFilzEUuUpe07s/xMDG0amlK6jj3Dh0+PI+h3psq8kiTs62lH/qUs6MZ34lS0vPuubhwCx5yyOakzaROCsl75eORHvXHdc98gLEpRVB0HAZx2gkd+NA598rRfQInNxniwKi+FbQrGQGFRIeqa6+BzRtcT7QT2zCxQX/wxPlVYIrbVb0OuJxclvhIIgoATdJ4Ht912G2677TYALDWlPSc3qtyRnw+xOTpo8lMdq0d3If0F+z2FSEXYyTHAgQWYRuoPIXOLxOx0iaRUM4NSF/sK8TmNVVt+COhSrjeVtLmukk7Z0QRlHNSThxTh/V9Mw8WT+iLT7QABkOGy4YLB2XjnkuNw8hB965KcDyqxLoAClKI11BJF39ECtzIcC1+XRwPpBbabwBOkm9FrXH2Q+Tet0FdEKiIgBjD3tbl4Y9cbiXVShu11djO4P2B8XLG1k1AuyW+RLe9o8x7F5mY0Pv88Wk34m19peAU3rLjBUv8CYgBPbnnS0j6xaKhiwWbBD1eplguSgGe3apPYvzr0Fc5961wAQO0DD8K/eUtcHZ5L+qP9HyXWyQ7GU2x+++3E9jeAVR5syzLmb1b68DgPVk871xDhMBqeekq1SMmD5f68l3e8jEUHF6XUFys2N2PKNop9L8UrI1EqoaxGRMfjbD60yXxgLn6gh5TqwVIKsbUVQkMD+hX48PuzRmDzPXPw7q1leOXGvrhrYh56SyZy9koShPq6KH67FqrbqqPzKVPALgFZfsDWph8hTEHRHGzGzsadlsaKiqIpHuyxCFMLLCHk54QQnWSzaViFTfZbZc6YYVh3ZOFIAEBxhnm/1tUjr8Y/T/4nDrYdVM2wYoQ/TP0Dfj3p1wAAOp2ZDcMTGHVHGdbPfY2+qVM02+r1wAMA9FU7jss/Di/OfxF5jjxLwvIcp5WdZnkfJbKL2NchGTtcs86csjmq2/960l9xzehrUBE4AOGnzLRJQ8G4egUe5qsbWzRWtZ0X57+I8wefr3n87bQSAOAfF29+ThUMqUIKcBlCe25uZBvnwU4vVU85aRbZp2mng/QMHx451vOnP4+TSk9CvWAcYGQFOX/5AwBgqCs+4K7nb34DABhhZ3MmYwKTdlSayrXA6W+pgEuWLYxdaDwOhXygCY1aZ2/Z9SRpR0p5HV4MyB0Ah82BkBiKbKdgaSPZD33OUP/s/shx5yAshnXrqcGelR3z+9hKKKEFs1+wxQDWEUJeJoTMJV1lA/shQQ6icPQw5jRyoXUryf5Ls0oxtCBxjtiA3AER8e/8PgOx5Owl6N2XJUiIygEqc2LtMblClfAMM+5HhjMDwwuGI8OWkZAebLJBNU438x8589XHw2FzoChD3dQ2MG8g+mX3A7URiIO1eaROO1t8cmUd0VgMLxiueQwAkSAUKbdrHi4lvhJLQgsOWeBBmeSeUziSEZggTiccRerj6Sovh7MP48jabXaM7DFS/5olCM9g5ueN9TcCgKusDO0eggy5jL9gKIO9tFCaVZqyXMREI6jKZXPB7XCDOBymBAjM8HftNju8Dm+c2ynblQ2vz1xAmtfpNVTcUQOx2+POgzidcbmIj0WYCnKilN5JCLkLwKkArgDwECHkZQBPUEqTE+L8gYK/dQa2GcuDcZNMS8iYr8axpmZNhD+bCFbsWwGX3YVCAMLWHeib3Rc1FcxELQXjebDBXdr0lqbXXjM8Xp2/Dh8d+AitYitEu3UebDLatwDQ0cIoNIEK9Sw7giRgU90m1bL3K9/HzoadcAoU9ne0ebDtYfYFX9lcqVr+ys5XMCx/GIYXqn/lOGS+I91jrCGcCGraa/DVYfNRm8E9zNSp5H/ylyOuj5sIaDgM/yb1ax2qqIjIwgWEAJbsWZIU51YLLe8y2UCuxaxE60cfwRegqJPLgjIHXDJhYl1TswbNQe2MZlYQG+TD0RZuYwFgAjWn9dponOktLIbRGm6FQIUoubpcTy4CTX6dbMYMlFI0BZsiLgQroKIIqSNeru67YCI2HUVMKaWEkEMADgEQAOQBeJUQ8gGlNHHh0R8oOA9Wjz/KsbWe8cw4784MXt7xMr6t/TaxzgF4YvMTER5s26pV+N/m/2HKF6yvNBDPg+WcSDU0PPE/w+Ptb9mP333xO2Tbs5HltP6FpsUtNYu2hkOMB7tV+2GtxbV9ZMMjaAu1wR0CHB+v09yfp4jTSubwuy9+h2tGX6O5wDrlbDy29Vs1j5EsrJjn/d+way42d774cd/a6mrjea2HjjVrNMu4378t3IZFXyxCkTf1X7AdDz8GANjXGh/3wPVgeRKKwLfsPjOzUKVqcQUAoVb9eaDMhEZDIdU6SogNDYZ1gmIQNW01cNgcUV+xgiSAyrmoeQS5GigoqtuqE1Y8kmKE42N/H6sw64O9iRDyNYC/AFgNYCSl9FoA4wGc24X9+96CW9m5uUsP3NxmpEeqhCiJcNldOKXPKUmnEaSlPfG3r/+GYA/mB4lKlSj/7exlbFp0muDK9nH1wfElx1vuIzejJwqnR+bBFmhrtfbwapPocz25OLGkM1exPTu+He4bs+JLV4LI+aulYmMN4aMBzuEm7ngerFZaSdNt99C+1hx8MS/PLcco76gu4cGSnipzVvZVVvdgc9/Rg9WxeZPjYicKm8sFBNuAj/4E/GUAhj00FUMePw2OLY8Dgn6QE6WKb09H/OIXK1f3zCPP4NH7H438VgY8cTeLElxCbsH8BQCYmyvLlWV9rEx4JZ9++mkMGjQIgwYNwtNPxwenAcCiRYvQu3fviJDA0qVLrfXDIsz6YPMBnEMpnUMpfYVSGgYASqkE4NgR3/sOgftPsuecalh3XDFLpN8r07x/TKISslxZ+Mcp/8DJfU9OrJO8rdmMtxmaMhpAdBIFmxxskHmyzjGcThT89KfImGgcQDMjawbumXKPpf557B6cM/AcS/vEIreYBbN4p6pz8xw2B84cqK2VOiBnAO498Y8AgJ6/vRvugQPj6vAXpUReIAAgW/YzZ8j6o6mGVT3YzBPZC4UygQj/ijm1n/G81oTTiRyNDD9KPVj+pXZa/9NwddHVKaXr2LxerBhN0HeeCvdZkrC9zIEeFzN94IwTmF6zmZeC1OrBsoQcdo8deHwWsPofQEc9CCgcgWY4diyG++Nr2eJrAEdREewqOYlj5ep8Tl9cfmZKAJqXA5oR78uNlZDLdGWib3Zf63qwedEBZLF6sA0NDbjnnnuwdu1afPnll7jnnnvQqGFR+OUvf4kNGzZgw4YNOF1HxzoVMLvADqCURtlKCCHPAgClNLkksGl0CQQqRCXm7lZIkqk30O86KI/C7KLrnuvIBQDkedIB/UAnTaer9GC1i6ihz/Go4vN/Ao17gZg8v0QKgbQcYOUJIlauLhb8SrSEmlV5sLEScnpfrlpydSETZu5ly5Zh9uzZyM/PR15eHmbPno33FbKG3QWzT4IopxAhxA5mHk4jQXAfQuPzLxjW/eQA05vUCo5RbZ9KCAgBTH9xOl7e8XJCfeSwvcSkwDzvM7+aMjm42NQEAGh+XYf3KIqo/89/0PqhcdL/JU1LsOCNBZb6lwoebH0VC24KLVupWi5IAp7Y9ITm/nua9+D8t9nXzqFFi+DfGO9n5UFnH1R+kFAfaTt7gDW+1jXJza3yYJuXsHkh1Hbm1+VBLC/t0NbONUQ4jPr//le1KIoHy2XTti3Grw/8OqU8WKm5GTO/pdi3WCV+gFIMrRQg/Jv5YluXs/EMHzT2XyfFg31yXtQ/1wc/A755EnTtf+MWVw4iBoF1T4B0NMTtz88FYGOo5Lcrcf311+PlF19Ga0srWsOtkZzaixcvxvwT52Pyuefh1NPOw/QJUzFmzBicd552xrOmYBO2N2xXHSsuV7dhw4aov512O5575unIgjtmzBiMP+UUTDr77MixDh48iD4Kd1tpaSkOaozHQw89hFGjRuHKK6/U/MpNFXQXWELIrwkhrQBGEUJa5H+tAGoBmL8T04gD58FmzzM2UXBuopWcwotOWIR7p92LxmCjaQULJR486UEsOmERAECayziNwamMv6nk1nE+WuasmZptlT78EGsnqB1BOKxgGN456x30cvVCnb/Ocn/PKD/D8j5K5MgmYjJZW9f2nEHqZuhHZz2K2ybchkpbAxquY3WoEP8AKcxgJuITeqmbod856x1cctwlmsc/5GABMnUT483PqYKW5q0aMqczU7VdYSLmNIy5/bV5rGaQe776Q5p4PPCO63SZvHXmW5hYPBFtknmlKTPI/x/zM47wxV/r3g/cDwAY6ikD0MkBN2MiHlc0zhLdTg+E02sCTfoV/TocYULg4guTxld7dnY2fvzjH+O9Z99jGtGyaX7hwoV459N3sHTV21j76qv47KPl2LBhg6q6DSEEA3MHIsuVZVktix+LL7gbNmzAV8s/wNpXX7WspHPttdeioqICGzZsQElJSZR/uSugG9JFKb0XwL2EkHsppb/u0p780CAHUZhJap7hZL4RvSi9WJRkliDTlXhKPb6YNwIo7NkfH13wEaSX30E93gGxKd7LZD+IPVv7PFz9jX1OHocHZTll8Ng8CfFgE02gz8ETlTs0zsNhc2iaZntn9kZIDEG0EwR7afM/+eKjNS5lOWW6feTJ/oXMrkkqX+IrQaHHPH+VC60rg964W4KLyycC4nTCnqt+rZ29e0eEJVx2FwbkDlDVbE0Wrr5s0VEmbYj0oVcvtHsI3HKZXX7hNMM57ZHRA72C5mMponBFjKi8JEHcuhXObf8DNETMAQDeAtCM/Pj9wRY+M/zdm395M8aNG4fzFna++CxevBj3/eU+lrIxZCxX53a4YQ+bM+crUy0Qux0vvPU2/vb4Y5FtNBwGRBGDZLm63r17Y5UiU1xVVRVmqCTxKVbki7766qsxf37XhhDpLrCEkOMopdsBvEIIiXu1p5Qmx434AYN/4fg3qvP9lOCm4aZgk+n239v7XkKcM45397wLt92NUgChDRvR03sTqrexJOVSMNjJg5VN3YFt6tQRKkloeNLYfHu4/TDer3wfTUKTZVOfX/AnLbrd3iRLj+1Q5yULkoBvDqtP9zd2vYH2cDu8AYrsZas0j8EzanFx9lg8veVpjO4xGmOKxqiWO8Lsy8G5W11PNlnUtNdYotcEdsrzQcH/5GO3uW5zwv2g4TA6vlYfz1BFBcJV7Pzr/HVYumcpqtpSfz2aF78IoDO9ZVTZ22/DF6A4LJcFd7PxFFuMM6Z9UvVJQqLjauD0GmnkQtjXP65qJqZ2F8jEn2i2QSXJlMxeZk4mFpyzAC888wLOuvgsAOyrcuHChQgcrAJtbNIVXKeUos5fF8kCZUWujooiLppzKi674fpIeWDXLtBgMCJXN2fOHPzmN7+JmHyXL1+Oe++9N67dmpoalJSwj4c33ngDI7pY7s7IB8u/nx9U+fdAF/brew++wOrxRzl2NDLSvpr4sxae2/ocXtn5SmKdA/D8tufx2i6WIKL988/x0PqH0PIV44FSBQeNi237v1mv3pAkmcq3fLDtIB746gE0CA1RPD6z0EoCYRZ8gXXt1E7ioMW1fXLLk3hv73vI8gO+b7SVUjgHkvOaY/HAVw/oLnAOWQ/WvSX1iRU4rJjnA/LLodjaucBy68OXh4zntR44x1YNPLnJofZDuP+r+7sk0YT/P+ylUG3x5jxY/sIb2MLyTovNTcbtpmhxBTpzQEtjrgby+gMxX9vU5gL19Qam3KjdCKWROAo9hMQQzrvqPDTUN8Rtp41sfzXTN5eQW7lyJUaUj8Dy5Sx/tVW5ulj3Eo35nZ+fj7vuugsTJ07ExIkTcffddyNfdl1cddVVkXb+7//+DyNHjsSoUaPw0Ucf4W8mtKqTgZGJ+Gr5/8nxPNKIAzezugYMMKzLeZOZFlRUBCog05GJBQMWYECu8TH0QMtK8Z+N/8G0kiFwVdVEmcL4366+GiLpivymzhJj01hvV28M6z0MlFJLqjR6HFUzcHnZtQ0Va0fo6skFZjgzMKP0JAAsSErNxMk1a62kI1SC82BDvY4NHqyztBT+DRtg83Q+WLmJOFkqihleNf9aHpo/FO4Od+p5sJIE0jd+zKn8Aljdm42nQ+6rzWuep55K2LLygas+ZNHC654A7aiH6MkB7X82hCGXwOvOBEzkIycqPNhYubqK2ooo7mxlSyV4aJFDhQfLJeSCQhC7m3Yj15MLSinC4bBpuTr/5s3RbimARerHvIhfeeWVuPLKK+PaevzxTslOJWWIo1UjuCsVMDIR65ILKaVJSGb8sMH9HlmnGL+7jC4ajee2PYeePmNxdg5REpHhzMCfpv0p4T5ySCdPBvAmQpNGwLVuS3Syf9n/5Js2TXVffjP2uPlmZIxTT3KvxJiMMbhmunU92PkDkvOl5BSVogVAxvHqeqgOmwOn9dcWFMhx5+D/xl+PPViJXvffD/eA+AWmwMsWxgk9J8SVmYHbmwsAyJqeXCJ9LVjVg/VNPh4t774Le17nywT3M5/cJ4l3cqcT2Rq+MaUeLP9antl3Jk4MnJhSug7xePD2CD/6zlbhBUsU64c40eNcJsyQMWECml99zVRe8VTrwdrz8juTmpz8G+Dk3+BAy36EpBBKDwmmk+I7e/aMuq+1kOPOic7PLN/fUkEuJI8LWiPA6TyZzkzkuHMiX7JmQOz2eD3YvFzNVJHHEozyVunxJSiA9AJ7jEKkIhwksbRkKQXXSLV9/3mwkSjMLjpXj42ZALsiNeB3ETwa1WFzIIjE4w1Uoaf5KkmQjgWKOUXyMsuRr9FE9WAZmoNNyAhnIteeq1GPq87+AJ4DCuhOE0rpFTr/4r/F0zANzoPl/hw9rNzHzI5WfE0iFeEX/Jjw3AQs3rY4sU7KsD33JgDA8w7j40bxYGX/S9PLGlxbeYGtfeBBtLxvLLL8cevHmLR4EjrCJnQsZaSCB1u3j31VhJeoc1QFScDjmx5XLQOAtlAbLnybfdFU33IrOtbH+6R5nt+lexNMzybzYOtfNOZOJwLLPNg3Wd3woc5AIB5Yp6edawiLPNintjyFm/ffnFIeLA0EcMZain1Pq/SDShi/PQw8zO7dFjndHg++0kNq9WAliA0NEfEDjtZQK4JCEFQQIJrkeYYP1UTd11qo99djW/22TjOx/P+8NsDWqnPPytUbg43YWr81rQerBCGkmBDyBCHkPfn3MEKIdmhaGobgPFgtvp8SJ5ayVIVWcgr/b87/cMfxdyAoBhPinT008yHcd+J9AADpHKaDGpzFUvzZFGYnezb7O3uBukmPeL3o+xRb/KiofVONKByBFeevQH93f/gFv2WqzrmDkkuJndebmXTJydq6thcNuUh1+5NznsRvT/gtKrOD2HNDNIFfiZ4ZzMR/Sp9TVNtZcf4KXDbsMs3j+93sdt03WcPfnSRcNpcl027WqSwdIs/FCzDqDKDNGTaLvEsuVt1uz82NpCYcVzwOH5z3AYbkDUGYWtcY1UPBq88BAEblDosr6yuLwQ/JZLENWacwDrij2NiFM7lkctJ5szl4Ok4aM9d8Tl/EXK6WEzsKdjtc/foZHivDmYHBeYPhcXiYUo+8YlIAjcXctKyd38rtcGNw3mD4nL64/pqBUnMYQJzJ+FiFWUPHUwCWAeCRBzsB/KIL+vPDgey0N+P34NF5Vky+hd5CTd1RM8j35Ef271HQF18u/BJlxccBiOaocR6sVoAHsdkivEU9uOwuFGUUwUXYA9rKS4HX4U2aC+lwsmts1zgPh82hyV8t8BYgz5MH0U7gz9VO+M4femrcSgAoyijS5S7zQA/B0zWm/wJvAbJdBg9kBWwZ7FyJvfMxwk2AajqqZkGcTth86tfBXlAQebi67W709PVMWeIGJZxFLGiOvzAo4SgoQLuHwCmX2bxsPInD2Aec485JmrPNoRaUBAB2YofD5mDlBpqphBDNdpSwERucdmckiI0vksW+YmR4jOcM3z8REzGx2yMc8M5ttu+PHiyAQkrpy4SQXwMApVQghBz73+fHMCI0nW/WwygmtKKJSe42Bs2n9XpmyzPI9yZ+I7+681W47W4MBBBY9xWKHTeiYRPjNkbxYGWaTmDTRgAL49qRAgHUPfSQ4fFq2mrwxu430Ciyc7TyBesX/Pii+gvT9dXQ1sjS/fm3blEtFyRBU67u+W3PI8edg5w2itLl2jSbliAzwe1o2AGUx5c/suERTCiegEklGoFWslxdxk7zknJWUNNeg5UH1FNFqiGwldGNxNZWOGVuITf9ra/VoG2ZAA2H0a4hVxeqqECoogLA37CneQ+WVy5HvV6mogTR+AgzDVe3VceVNTz7HHwBGikLbGc0OqVsnxaWVy5HqjIZc5pOLLhuNBUoJIMIWSqKmrJ3SoTEEJqCTXH3Za4rB8EjVYZnxPdPhIJHRRFiWyuc6MxkJ7a2fX9MxADaCSEFkG0AhJDJAI79EK5jGJFEEzp8Pw6emIDnADWDp7Y8hVUHViXSNQDA67tex5K9LPNLx7p1+POXf0bLBkb+j+LByn9rJcyQ/H60LH3P8Hg17TV49NtH0SCwc7RqIuZc4UTR0cz8WK4K7cVLi2v74o4XsWL/CuS1ATk7tEXum0PsltGKIn3020ex7rC2nqxNnjOZu7WPkSxaTdA5OAJbmc6H1NYphs3HLZkFFgACGoLrSuxp2oOHNzxs6b4wi+CLjANe3R6/wDY+x8zHrfJCFpSTk4gtxo/EVMoExPpeVY+nkrIzqlySILZqvxhwubqQGMKRjiN4+O8P4+G/PBw5j4AQAJVfLNQsCVyu7owFZ+BIxxGEpcRM+bG6tjQcLwBgRq7ulVdewfDhw2Gz2SLc2K6E2QX2ZgBvAygnhKwG8AyAn3dZr34A4OY+95AhhnVLM5m2phUzqEhFuO1uXDjkQgzJNz6GHuigMjy37TkE+zCzmTK1Gv/bPVDlkwyI8kVqcmUVKHGW4PzB51s2++lxVM3A7WN+sZBOqkM9bqeN2DC7L/PF2XsUwlEQb5fgZtO+WYn5UG0Odq3be6fGxJgsXGVlADpNxUAnD3Zw3uCUtK0HgbLFY1SPUZiWOS1lEapKH6Gtf3w/eHl1GbsfnfK8tvsSN4snA5uck3jJniU49dVTcfGSi3HDyhvwfv1nxipWinNVS/UYK1fnIA54Hd7Ita5UBGw53PHulVi5Oo/dgzxPnuWxijUHx/42K1c3YsQIvP7665jeRVS3WJhaYOWUiCcBmALgZwCGU0rj5ULSMA2+MGVOO9Gw7rBCFmhhRahbkAT4nD7cOfnOhPVHOaQTGW8zPG4ogGhhaa4Nm3H8ZI2dmUmo52/vhnfkSMNjDXAPwN0n3G0pEMRj9ySnPwogu5CZnzImqItEOYgDs/rO0m3j6hFXAQBKfv971cCRfDdbGLVSIRqButmDNP/EGQntbwSrerD8WinzaXMe7Im9jee1JpzOSABVLJR6sNxPf1LpSbig4ILU8WDlReelaTb0OUVFjEOS8PkoF3qcwdL8eceMAcD8w0ZIqR4sAEdhIexZWViyZwkWfb4INe01oGBpCf+0/zEs95vLZuvs3VtVMD5Wrs5ldyHLlQW7zR71IiL2yIWoEhsQK1fnc/rQK7OX6lhpydWFRTEuqMmekxO1yJqVqxs6dCiGmPioSRUSTTQxmBCSTjRxDEOiEuzEDkFiurDdpQ2biEaqRCUQEEuZnI4FcJ9QXNaZFIFfjxKfeVWl7zO4OZoQApGKlrN/aUKes3o8WM2yLsQV718R3Y32dpzaZxYW9vwp/v713xEQo3MRB6QQ7t/xKM6c8GM0BZtw4+rolIlPzjVHbbv++usxctRInPtTFqlPKQWlFIsXL8Yf7vsDXAIg2gGbwwk7sWsm+wdYEBxfmGPHikvUASwXMf87sG0bnn/9dd1k/1bk6o4mjJ4EC+R/PwHwBFgUy0IAjwNI82CTAPdd1j+urTHKsbySZT3RShKvBs6DHfvsWDy39bnEOinD/iS7WbxvsAAYZQYVzrNrXKzBtZVvpkOLFkX0Q/XwTcc3GP3MaEtcwVTwYI/sY3404W11rq5ABTy26THVMo6rljHm2oGf/gwdKr71A20HAABvV7ydUB9t7czXeeRZdf9SsrDKg216lfkpwzXxPNj/bVbRUTULszxY+Qv2yc1P4hf7f2HZb68JeYG96BMJ+554JK6YUoqp34bg/KfMg32bjWdov3Yea45U8mABJrQQPnIEhzsOq5Y3CwY+dfn+DB88qMmDzc7OxsULL8bixxbDL/hxxM98qZcsvARvrHwNa199FV+99CpWf/i+plwdB+fBWhkrKoq4YMZJUXJ1a156CWsSkKs72jDKRXwFABBClgMYRimtkX+XgFF30kgQNjfzMeZfdqlh3Zl9Z+KDfR+gLLvMdPsrz18Jv+CPJOy3isdPfRyEEFRiHMSFZ4JgKQLzT4Lvv6/CpuDWcfNg7gUXqLbj6NEDZa+8jMrz1cs5RvUYhTWXrMG/l/8bX7R9YflhefFx6rxJsyjsOwStAGxztXmglw5TH6sX570IG7FhZtXJ+PLnkzDpX6tU63FfulbKxTWXrImYWNUgyeb4zVN7oSuMXDnuHJxQop0fNq7+mWfAv2EDnMXxPNgfDftRUn3Jv0ydD+zoVQLPYHb28wfMx8x+M7F4a3KJVOLgdKLH0tdx5PRzMLZHfHrP8nffwaZJEzAwl6VszJ43D+2ffwFXb+M4gBmlM7C9UV2xyQixX5xUEBDYztrq6euJmvb44LeeHjY2ue5c1S9W4nLB1b8/Qnv1F/3bbr4N48ePx0U/ZlxwCvYF+8f7/ggHscMWFnXl6uw2O47LP860IljU1y0heHnlSvztR51ziobDoKKIwRbl6o42zNJ0+vDFVcZhAF3Ddv+hQJ5AZjQkHTY2THZi3seU6cpMKmKRa9ACQHFuKTZethENzzyLw4iZ/LI5VOs8iM0W8dPqwWFzsH8y19fKAut1eJPmQtpkn5DWeThsDlVOJNB5rWx2B4JubaMQv258PGNhljsq2rvGPpnhyLB0HSPXSmES58ErVrSL1drV0ii1eTMiQuNOuzOKm5kqEEJgz2RjoeYrtPl8CDtJpCzCIzXhGnA73BHRh6ShON5N427Cos8XRZmJPTYXbii/XLcJQogpl0ZBQQEuuOACLH5qMRZcvACgTK5uwXkLYBclOPZW68rVEZCoa2lFro7YbLjk3PNw2c8742rD1dUQm5vhGcriQszK1R1tmJ2ZKwghywghlxNCLgewBMCHXdet7z9omIWrt681IVfXwCgodQFzUmKUUjyw7oGkuKHPbX0Or+5kb6Htn30GABGNTklJ02ln6dH836jrd4pNTTj85z8bHu9A6wHcv+5+1AuM02gl0YRf8CdFSQKAlnr2/ujfqB67J0gCPq36VLXsiU1P4O2Kt1HcSDHyRW1d2qZAEwBgS5061/b+dffjs4Ofae5PZKpCng4VKBnUtNfg/cr4wBAt+L9l10qpg8p5sMnI1dFwGG2ffKJapjQRf3P4G/z1q7+mVAIOYJSQhvuZjNmB1gNx5Uf+9RB8AYoqucy/WZarMyH7lioTMaUUgiJF5bwB87BoyiKU+EpAQFDoLcRv+l6NORnqQXuRdsLhqFSXWggKQfz4mh+jXtaOpaCwERtynFlw1mnTfLhc3YoVK9Crdy98uJwtG1bk6qgoxlGJxJaWKB6sWbm6N954A6Wlpfjiiy8wb948zJkzx/Dck4GpL1hK6Q1ywBOXTPkvpfSNruvW9x98cpjh+1W2VALo1BM1gkhFPL31ad20e0ZYuncpst3ZGA7A/+23+O3nv8VFm78FEK3FSEPs78A2dbOX2NqK9o/VH5ZKHOk4gme2PoOTs5iJllMwzIJfo0QRaG2EG4Brn/bDRotr+1bFWxicNxiFrUBulfYYtYbZQqSVU/qZrc8gw5mhGYFL5JeynP3mE45YBfehmqq7axcAFmzDwS0PyQiuA0BwhzGveUv9Fjy55cmk5rkaaDiM0DtsEVfzbTa+yMTY28PsvEMVsuB6F8qeqUFoiOb/zhswD/MGzIu8wA04RDuDDDVARTFq/GLB5erCUhi2bBv2HNmDen89KCgkKsEfaodN1gP22OMzlHG5usZAI6rbqlHgLUC9v96yXB3/IIn0W4Xfa0au7uyzz477au5KuTrTthVK6euU0l/K/76TiyshZICcU/lV+beNEPJHQsi/CCGpvUuN+iKHmHtGjDCs2y+bUT7MUld4thSPw4MrRlyB4YXDE+yl3N6wgXh91+sIDmCZMqN5sMyk6D5Owyso3+D2ggK4+xvTEwqdhbh8+OUo9BpLfylhxT+tBk9mLgAg1FebCjUkT9/zeXLvGaytESNU00NmORldYUBOYvq8VDbJNvbJTWh/M7Bi2nUPYrlwbT6FfKFsrh1ekNyccw825tFyK8e44nGYlT0r5TxYCYBjyKD4CvKcPjiYfSG5BjAOuFlpuFRDjV7DYTaiXc8sr4TL7kKBtwAO4oBIRVS1HJCPY4fDY2z69jq8KPAW4L33jZPPRPXP4Yz5fQwohZmA7tUnhHwm/7+VENKi+NdKCNHNC0YI8RBCviSEfEsI2UIIuSfRThJC/kcIqSWExL0WE0LmEkJ2EEJ2E0Ju12uHUrqHUqoUKTgTQCmAMABjKYwUgvuvfJONOao8UYRZUXFupvM5fbh5/M0YX6xvJjICPX4MACA8ij1sonmw7OGaMV5d45S/QRfffjs8w+ITp8eiyFGEWybcYilxhMfuSU5/FEBWATNXZcicxlg4iAPTS/XJ6fP7M85k8a9vh6u0NK48150LgAkbJAIqPwB7TZ2Z0P5GKPGV6GrexsI7ejSA6ITyPEgrKe6104lMjQAVJQ+WWzmm9JqCM/POTB0PVp6zT8+yofeJs1XLV0x0o2guE7jwjGAvE0pdXC2kjAcrvwQ4iophz4zO25zlyoLb4QZxOGDLMfdS7uzZM5KwQg88/7PT7ozK7R8qzEbYrX39eTxIhiMDPX09LY0VsdvjRAvs2dnfiVzERq83CwGAUppFKc1W/MuilBpleA4COIVSOhrAGABz5RSLERBCigghWTHbBqq09RSAubEbCSF2AA8DOA3AMAAXy0o/Iwkh78b8U8s4PwTA55TSmwFca3A+qYV8gxilMgM639TN5vHkZjobsaE52GzJ7KcKQeYbyv+PUsOInIdGCjT+NRDwx5l5VKuDoi3UZimlWkAMJJyCjUPi11hjPAQqGB4jEGa+QMkfUM2TKoGNX7Kyar085hOOWEFQDCIkxaeg0wKV5wMUZkj+IE3qHMNhzblCA4FI2jx+PwiSgA6xIyGVFlXI5+MOA2Iw/t6hlMIeEiDyWAQx/jpoISSGEBRSqF1LpTgzMAXjqYL/091fvn9lbqsZiBLjHCuDKJsDjfCHtOXqlG0LkmBprKgkxdWnlIJKqUs72VUwWmAjpmBCiCW+B2Vok3865X+xV+QkAG8SQtzyMa4G8C+Vtj4BoJZwdBKA3fKXaQjAiwDOpJRuopTOj/lXq7J/FQDu0DqqmaMl+cZteMqY07hi/woA5vVg+YLsF/w48cUT8dL2lxLsJYP9WTYNPO98DACQlHqwcmBH00v6erCH7robLcuXGx5rb3AvTnjhBKyr0c7Jq4bntiXH9a0/IPsTl67QrPPUlqd027hvDYtaPHDVVfB/+21cebJ6sJwHW/t8ElqrOmgINOC9veZNd81vvgkACB/uvLVCIlv8ntlqrHOs25cn1XnN4YMH0bZS5mPL8/zZbc/iV1W/ShkPlj/ML/lYQtVzKjx1ScKMb0V4/svmfIvM7w6Z0IP96MBHqvmNE4Vw5AiE+mixg7ZQG0JiiAUHmQi8ApiWrZEwAMD8ztsbtqND6IhaYAtbAFubsYZzY7AROxp2WBsrSiE2Rj/+xcZGIAHhgKMNowVW6dSw7DgihNgJIRsA1AL4gFIaJUdCKX0FTAbvJULIQrDkFedbOERvAMowvyp5m1Z/Cggh/wYwVlYGeh3AHELIvwCoRuIQQhYQQv7b3GwuwMgsuDmm4CpjWd05ZSzSrTxHI99vDHI9ufj20m+T4oY+P+95/HvWvwEA9IrzkenMRPAclipQmRqPm8XyFsYr6QBMs3LAkncNjzeueBw2XbYJw7zMjGwlyMnr8OLy4Zebrq+GojJ2XPsC9RR9DpsDV428SrXs7bPexgMnPYCKodl494Zxmsfger4Lyheolm+6bBOuH3O95v6cB/vFlFzNOsnAaqrE3PNYZh9nSWc0KKf5XDki8Tw0xOlEwdVXq5YpTcTXjL4G63+8HjbzoSSm4MjLQ8+PWcKRCcXxro8hX61Dh5tgSD6Tb8w5k10zlyKTkBZSZSImNpumyyXblR0xERuZrW1eL9zlxs+VTFcmhhcOj1DJ2BcyINiB9r7G8RL5nnwMKxhmiWrIQex2OPKj01A68vO/FyZiqvG3KVBKRUrpGDA/5yRCSJzziVL6FwABAI8COEPx1ZtyUErrKaXXUErLKaX3Uko7KKU/oZT+nFL6sMY+71BKf5pj0pdxrCCV6RF7ZBThi0u+QP/sBB8MFtLX8YdlIrJW3Q0HcUBKoVqKFqRUmUK/4yCEaHKKk29cv7g7UiUeC1AGkjlsDvTJ6mOK10vIdy/1aSpg9AQezYOaAIyyEuSkBKW0CcBHUPejTgMwAswc/VvzXQcAHASgfG0slbcd8+B+pLbPtPVDObbUs7D7Wr+alTsezcFm/O6L32FD7YaE+/f4pscjKRbbPvoIACIanZK/k3fIQ/w71qprpYZralBz992Gx9vXsg+LPl+EIwLTprTKg/1g3wem66uh+QibNv5v1GXWBEmImOpj8dD6h/DKzlfQqyaEk17YpnkMLqu28Yg613bR54vw0f6PNPcnsluhZIc5PrRVWE2V2PE1SwepTJ3J/dSrq43ntRZoOIzWD9Vp9koe7Af7PsCfvzTmWFuF2NKC+nv+CECd/nXod7+DL0Cxr5mVdcg5c8UGNS9WNFLGg5UkhKvVTc0toRYEhSCoIBiaiKVQSLMdoFOuLigEUd1WjQcfeDAiV2e32ZFp88J+WFuPl8vVzT19rqq2rhlQUYyaYwCbc7FxDmbk6hoaGjB79mwMGjQIs2fPjiSmWLVqFXJyciICA7/73e8S6mssdBdYSqldEdTksBLkRAjpQQjJlf/2ApgNYHtMnbEA/gsWzXsFgAJCyB8s9H8dgEGEkP6EEBeAi8Bk9Y558MCE4HbjtGlVrcy3Y1arszXUild2voJ9LfsS7t/K/SvxWTVLehDYuhW3fnwr2nYwgW2lNiP/O7hbPU+y0NAA/1fayRc46v31eG3Xa2gV2Tla5cFy/2aiCLazG9h1UFt8WuvBuHzfcqytWYv8JgE59QHVOkAnb1JrXF7b9Rq2NmzV3J/IAVh51V1m5LEEnl5P6lC8cMmWB54cJdm29bC+dj1e35V6vRGpowPhFcxjVOePf5lpeoUlYOFZk8L72HiKOnzSlEOSzPlXDawdNCxEvTDHgsvV1dTWoDHQGAnUA9hLcEewFVRONqPGg+VydSIV0RhInL9NRSHmd/Tialau7r777sPMmTOxa9cuzJw5E/fdd1+kbNq0aZFcx3eb+Cgwg66UWCkB8BEhZCPYQvgBpTTWGZcB4AJKaQWlVAJwKYC4pw8h5AUAXwAYQgipIoT8BAAopQKAG8D8uNsAvEwpVU+Tc4yB+w+8Y+NzncaC8ybzPeZ0QHkAgdfhxQ1jbkhYHo1DGj0UyyqXITCYZcck7s50ejynsmekBvVEvr/dw4bCPVCFUxiDTHsmrht9nWl/M0ey+qPebHZtQzLXVw0jCvTpNSf0ZMT5zFNOgVMlS022i72TJtpXzoM93C/ToGbi4FxdM3APZT5IW2ZnikfuY0t2zpnhh4uSCLvNjgk9J2BezrzUpUyUF6X6LMA5It7PyZes6uEsmts9iI2nPfvou5GI3Q5bBqPKhQ4cQMW8+SDTLwD50S8QPnwYxG5kQmdnY/NmqPJgY+Xq7MQOn9MHl82FoBhETRvLKkbcbji88XqwsXJ1GY4MFGUUqY6VllxdKByO61tsSlOzcnVvvfUWLpPzXF922WV4Uw7U6yp0GVtX1ovVXT0opatjfocBxEmWUEo1o3UopUsBJBaW2Y3gEyRjvHZQDMfAPMZcsrrAZjgz8LPBP0uwh52g44YD2AVhWH/goy+j+HJEvrm9o0Zr7MzeeHvceCM8Q4wXlkxbJq4Zc42l/nnsHkztNdXSPnHHzStCC4CMEeqatQ7iwAm99BPhH99zIg7iOfS46UY4e8Uv1Dku9gA+Tg6OsQrOgy2fqqJRmgKU+Eowqeck0/W9w4ejCdEJFrhPdFyR8bzWhNMJ35QpqkWu8nK4B7EXNZGKcBAHxhePR2tua+oWWNm69NJ0GyYfP0O1/N1pHvSYyYIP+YuGPdd4gZ1TNgc7G3cm1K19P+4Um6CUQuroQPZpc1F49dWQ/H7sWXAGaCDAvKSVVai++RYU3ngjPEOPg9jUhH3XRQfQ9Xu2M9LbUVykyYNVytXZiA0+pw9uhxv/e/p/uP/+v8ApANRhh83uACFEU66OgMDr9MLrVPfZ6snVvbT8A/xNISiSqFzd4cOHUVLC5B579uyJw4c7M3V98cUXGD16NHr16oUHHngAw4cnlywF6MIFNg0DcH5oh7Z5hoNTH8xyC7n/khCCmrYaZLmykOlK4qvHz0xhJMD6EcVJ49qZAY3zkMvF+npIgYAhmZ2C4nD7YWS5sqIEB/QQEAMR82uiEGUer6hxHgIVDI/RHmSmW6GuDrR//7i3bm72jtXtNA35uvdyGAt7J4LGQCPawubNz1KA+YSpGM+DTSo/cDismb5PbGyM0MREyr5gW0OtaBAaUqYHy+d3TjsQbFMJNZEkeFqDaGhiJkjKr4MJTntrqDWSkzolEKWIuZQGoucVDQaNebC8riCASpJq5ielXF1JbglESYQoibjwkgsxbc7xKK2jqMsGsnr0Qq5X5yOAsGeTSEU4bU7TY0VFERefdSYuu6HzBSFUVQWxqQleE5YOze4oAq/GjRuHffv2ITMzE0uXLsVZZ52FXXIq0GTQPSrcafx/e+cdHkd1tfF3ZvuupFVvVrXce2/gXnEBGxtjMBgMJMGmhhJIqIEQAoEECC1f6B2bYjA2xgUX3LtxL5LVe1tJq+0z3x93ZrVlZmdWWlsG5vc8PMg77U69M/ec97xeHWzDShH9qA/bS0kstLhZ2m8SIHEwg9oAl8eFaV9Oa1ecSq/We2Mq9BdEG6nbQMwD/HSwFvK35RuR0Dd3AVc88iiaN4lrTGmKhlFtRJOnCVO+mBKWHhMAVpyRPo6haCjj4ombxIvtf3LqE8Hf9So9dCodPjpNppfcehtsx4Jr8VZaSZ1jsWQpo9oY0q6ObiWxrqovO6ZrFsPusYu2TQg+2chdE6yD5Y0i2ouYv7Cnvh7WnTsBkOFKg9qAT05+gifKnoicHyzH4i0Mqld8GvQ7bTBgymEWUR+sBgDvde2ukDZh2Fm+Ew2O9sUisz/8oO2/d99B2tNPwXT5ZXDX14M2GKDNy/M67LAUBU1GhrdSnCo21n95n69XgNPBtoi/XN1x1x346uOv0NzSjDp7HZqdzfj8k89x5ZT5GLlgAWZNW4Bxw8di0KBBWLBggeA6aIpGo6MRZxvOSp6rwM73448+8g4ZDxo0CMOmTMHIBQu82+rSpQtKStoUm6WlpegiYB+YkpKCCu48VVRUIJkraRoTE4MoriLWzJkz4XK5UFvb8WRCpYPtJLw6WBHfS18mZ5PSeHJlMj3je2Lv4r2YkDmh3e17Z/o7eGniSwAAasl8JBuT4b6StMNfBxsLAIi99lrB9RgGDEDXtdJG64OSB2HP4j3opSfDbeHa1XVUB5uUQ7arnjlFcHooHeyKOSvwzOXPoHBwCt7/XY7oNjKiSfnEmbnCQ7x7Fu/B7wf8XnR5Xge7ftCFkTuEq4M1zyXz+sabeR3sjX1ubHc75OpgHx31KNZeHfnokDYjA6nbiA52SHJwlKvnwQOw6n10sLNnAQA0AuUxA4mYDlaj8Vq18WS++Qa0ubkATYPKyUDKo49I6mBVJpMsHWxmaiYWL1qMj98nLz4sWCy8biG+2PIlftq/HXu++ALbN68XNVzXqXR+oZFH/vIIvv5avKQ9b1cHkDjzDUtu8jNcP/Djj9j79dfebU2fPh3r169HQ0MDGhoasH79ekGnnCuvvNKbYfz+++/jKk7DXFlZ6R252Lt3LxiGQUJCx0eKlA5WQZJEQxI2XbPJazpwIeHjaJH+GrkYqChV5Mr1hYDFL08jrHDh0WZmIm/Nd4javQ6aT18XTLTrCPfff7/fV51erUd2TDa0tLiHsK9dXUZGBn7cQKpwhWNXJwe5dnUPP/wwNmzYgO7du2Pjxo14+GFSvv6LL75Av379MHDgQNx999347LPPIhJuUDrYToKXtzRv3iI5L6+bFLLOEuK85Twe2vYQzjUKS2fk8J9D/8HbR0mZuGauxGHLdmI9xbS2lUTjY2W8Z2wgjvx8lD3wgKw2/2nbn1DlIvsYrg52bUHHvmQs1Zy35z7hEo1uxo1154W9Ul/Y9wI+Pvkxupyz4JpPxcvl1dmIXvBg1UHB6X/a9iesLxQvJ0lxsb6upy+MvVa4Oljey9hXLsLrYDviz8u6XGj6XjhE4KuD/fjkx/jX/n+1eztiuKqqUPvQowCAggBpFut2o+z+B2CyszjfSEqXtnIyNHetuB6UJ2I6WLcbLpHSjH46WAGpii+M3R6yxCNvV2d32+EyulDTWIM7/nQHWLBQ02oYWQ2oCvHn0k8//YSamhqUN5Rj36l9mDR1EgDIsqs7deoUhg0bxpV89N8Pd2NjkFTnlltuwblz53Du3DksXbrU+/tbb72FYcNIRa6EhARs2rQJZ8+excaNG72d8J133onjx4/jyJEj2L17N8aIJNmFi9LBdhK8DtaZny85b4WVxAzkJqDU2mqx9vxa1Nukhe9i7CrfhX1VpLNxnD2LZRuXoSWfBP19C7HzLwrOImFtp7u2Do4T4sUXeBrsDfj+/PdoYcg+hvsFK7cIhxiOVtJpaSvFH0ilLcIPom1l23Co+hDM9Q6YrOLtbnWTFxMxze7357/H2UbxxAqK0wIm1ESwWHwHcHExL8bWllzD62Dl1s0WXbeMur4Hqg5gW6m013C4ME1NcO8i136D3f8eYhmmrfYwZ4zgKiNtZWzStXgjBcsw8DQJJGCFux6Xy8/fWQw344bFYWm7L1nA5XGhxW4By51/g0q8olOruxVNzrb2rl0X3gtxkK+tDGOFSwGlg+0keD9D4/DhkvN2jyOyhAS9vJgA/5AzaAx4aPhDGJ4qvY2Q6xvWH9vLtsPRh8SOKJ9MYD6WLKrn5doSNXFiUMxICA2lwQPDHsCwVGH7OzE66j9qNJN6qk5O6yuElPRkMBevi7v+OmgEEix4P9/eCdLHQQheplOUJW0r1l7kSsEAQN+fSJpU0W0Z6rwOtqPXXCh9OC9x43WwY9LHYH7c/IjJdPhh/jPpgHbwAP+J3IO9LhooH0xirnxNYJU5VnLdfIw6UtA6HVQmk+j0QL2oGKqYGD99u+j2KBopphQYNAa0ultR3UoKs9BRUVAZQ2T9s0SmY9KYkGpKDftcBbZNjnftpYAi0+kk+A7WMHCAxJzwJkXE6aX9JoG24VWj2ogb+tzQzha2wfbvCeAk3D2ygA27vMUlAIDivGFFvV65B1LCbbdC11XaL0IFFW7qK5345YtepQ9LvymEKTaR6GB7CXd+vN4yFD3N3VEBIOHWW6FJCbaUi9GQQhPdY6ULbgjBF5oYNPbqdi0vRbg6WF7XTPv4kfI62P6JwnpiWWg0oi+egTpYFaVC/6T+qIupi3ihiW9H0Zg0JEBfzV3Pm0YakDSWDHdquxGdusos5eAJTMicEJYOVlR6xLVRlZTkLTTBE62NhpNxglK7/c5NKNSJiX73tRgURSHRwL2MepyguJQDa7QGRjUg9urHkh6WqBPU4b0gUioVVCb//VBFRQWVT7zQtCe/QvmC7Sx4fahFepjHxvmM8hIIKXzLDJ5rONdx3V0TGbalWkg7/IZruDgI0yy8H7xno7OoGJ4QMgBfCiwF3nilHOwee7ulDzxuJxkmc4nsh5t1o9HRGHIdzQ6yrLO4GIw9WOvqYsnQejhaUz9483pWfrWlcKiwVqDOLv+48+fTV//Jj574DgeGjcslWgbQVV4Ody35anKzbqgoFert9ahwVkTcDza1AbDV+5fO5K/n2Do7WqtJ6IZpIXkIrEtaB1trq/VWP5JCr9ejrq4u5H6xDkeQ/tbNuuHyuMjvUtpcXo/vdAp6GAvhcDuIp6uPsYXV2gB7CD9YgHzBuhk37G57eH6wHo9gqUS57Y0ELMuirq4Oehmm9L4oX7CdBMPFLi1rvkPy/feFnHdvJUkmKW8plzW8qKbViNfHw8W4MO/beXhw2INY0neJ5HK+xOpiEa0lD3J67RagB6D7iSTnMC0tAJcc4OE8JJs2bEDismDPev5LveIvfwGl08I8a5bg9jS0BkmGJKgpNa5adRVuH3h7SOu2QFadW4WnL3ta9vyBWKpLiGHxDnEf2i/OfIHHRwfXKI3TxSFGG4ONFVswCkDx0luQ/cnHMA7xH1Ku4YbTdpbtFLQSTDIkwaQWH+6juVhXxZqvkbR8uYy9Cp/tZeI64EBathJ/YE9dHcBV0eGTnNaeX4s/Dv1ju9vRuGIF0p76a9DvrM3mrW0drYkGBQorT6/EqxWv4hr2GqipCDzSaBWg0eCGzS6sif4GuOxe3xZAlZSIaYdqsVa1AZgNWHeQgnTu6irouoaW4Byokq7LzZORkYHS0lLU1ATXx2bdbrhraoCqKtDR0X7VtPii+q4mAFVV0LS2wm63C3YOjN1OTAqqqqCKjxctBOPwONBob4Rb70atrRbR2mioKBWsrY2wWgFUAbWVzdBHxwou3+hoJHFctQUWhyWsYWJXVRXZD58XdN6gQBMByzqxYxOIXq9Hhgwpli9KB9tJ8Bdy/PXXS847IXMCviv4TrZMZkz6GGy9dqtscwAhXp/yOgDgJL4Fdd1VyDUfhGfWSKj/8xFUMW1DYarYWABA7NXzBddjGjUSXdeuQcFM4Y6Vp39Sf/y48Eds2bIFakodVhaxQW3AtT2FdbhySczsAQsAzfRJgtPVtFpUa/v+FURX9zfqb3jZcRr3fNAoOF96FCmfODVnquD0Hxf+GLKNjIkMBa7q2QTpwEL4hDtEbJ41C627dkPtMxzOxxgX9VzU7nZQGg3ib75ZcJrvEPE/x/8TAPDfI/9t97aE0PfsgdRN36Fy3HQMSPI/0qqoKPT46SfsH9THq4ONmTEdLZs3C5bHDCScUokajQa5ueIdNuNw4PTAQUi67z4k/r5NN/x/m/8PhU2FeP7FekSNH4e0p5/Gli1bMFgkrm0/eRLn512NjFf/g+gpwjpwHg/jwaAPB+GOQXcg2ZCMJw49gfcGPAfjdfej7o/XYOIfQrvQfHLyEzx76Flsu3ab7JDX6ZuXwjxnDlIffcT7W+XKL9C0di167N4lax2hCHVsOooyRKwgSaIhEd/O/RZZ0eIJQJGEpuiw3XQuBWiKDuvFoL14foFeuQq/Dnx16uMzxuOjmR8hVh/buY26hFE62E6Cl7c0rZf2MeV1k7xcR4p9lftw1493obq1/dKVf+77J14/TL5im74jJkgtW4hXqW+dWL68WsuWLYLrsR05ghKBoeNA8hvzceemO1HqLIWKVoEJIw3f5rZh1blVsucXorGKlKG07xL2tXUzbqzOXy047eldT+Pto28j/XA57vpUfNSAtz7bUyG8jTs33YnvCgINp9qguLjugNMXRqYTrg62hR8a9fFB5aUrHfHnZV0uWL4RboevDvZf+//lvUYjiaPgPGruJGGbQC25p7kZJbcvg8nO4lwDkVRZd5Pz6ZZRWi9SOlhnaRlKRe6rjcUbca7xHNzV1V5rPTFaDxxAyTLpcMOp+lO4c9OdKLAUQEWp4GE8SDAkoEe1Gu77pUMzbxx+Ay/se0FyPiEYiwWNn/uXB234/HN5dn2djNLBdhJ8opCrWLq+cI2NxGB4HaUUldZKbCnZ0v6i8iBemz/XkgIXzqIi3PT9TWgpItpG36QKXhPrEnCuAMhDx1UkvY8WhwVbS7eixdNChojD1MFKJSBJ4bSRFwVNrXhmolihj31V+3Cy/iSi6lqhCdFsvgC+2IvP1tKtIT18KS6pI77x0qhy5a4gtZVZh48/MJe8UtJcIriM7HVXS78c7q/a771GI4nH0gjPz8T10uLwvx5Yp9P7Mslfo+4qchxC+apGGqalGdadHR8edVdVwV1ZKTlfg70BW0u3wuKwkA6W9eBsw1n8dPQ7MAWFAEJnjh+uOYxDNYfa3U5f7T0Ab3LlpY7SwXYSfPKPmC2XL30SiAQmyZAka938jW9Sm/DUmKckbdakYEYNxsHqg3AMILEvSt8mKKc5mY6YrIJ/kYhddK1s54uHRz6M6TnBdURDMVigZmw4RMWTOKKrj7iUaGTayJDr6BVLZCtJf/wjtD7WWTxxOhJzaq+EhdWS+Oap7I4ndoiRZkqTPS+vVfWVp/AynTHpHauEYxwpfqxpLqHHzbihptQYnzke1ydcH3GZzu6eFPQjAvTY3PV8Ng0oH0Hio/r+JE6rlqj7C8CbONhhuHaYxo5F1OX+UiKDuu3+1GSHDuvwWdHxNy2Brpc8G8VHRz2KKVlTsLN8Jz46TkwDzFdfjejB4tp1D0MkVSPTRuKpMU/Jdsri0fXxT+7U9ewZ1vKdhZLk1EnwHay+j3RWMF8kni9UIAUfB9Sr9ZjXfV47W9gG2zsPwFG4u5LiCbSuTeTNF53Q9RDxeuUeVnHXXQdttrwkrSvzrgyrfXqVHoOSBoW1TCCG6Di4ABhETOHVlBoDEkOnFqWb0lAFIO66RX6JYDxRGqLla2+xd1ZDrpnx48PLCJdLuElOujzyMuKrw+QLTbTX8xYAoNHAMFDYX9g3yYlhGdAUjV7xvTA6anTE/WDXD6Ewu7//iyP/hb5riBFJo8aSNuXmAGjr+EMxJn1Mu/1ghdoRd92iIA36mPQxKG4uhjq5DqYREueTW0/sokXQysyQ5Z8pfKU3APhhMDDSbIOIGt6rWc6LzUNerLS5gC+02QzjEH8NunHoULirhEeULiWUL9jOgruJ5dQvbeL0lXKHfH2HV3+u+dkrD2k3dURjSjeQ+KKf/owbLvb4xOH84N6Q7SdOyo6ZnK4/HdYQo91jl12nWQyXg9MaNwifDzfrloxp8zpY24kTgn6mfHwycNhRLhR3zSQ5L0wVmwprhew4P9BWg9h3+I7XwYajpw3C5RIdInbm53vLKHpYD9S0GlXWKhQ6CiOmg+VHXXKrgOaqgJKN3DZSKmxoKSkEQGKEQFteRSjKmssiEoP1va/cATKe0uZSnG88D3d1NVzlEueTO1+Okydll148VX8Kpc2l5AWHO+RHdqxCftFh0WX4DrbWVoufa372yrnkwFgs3mF4HldlpRKDVRCH8Rb7Dy3NAHyK/VvldSJGjRGZ0ZlwMS4sXrs4bG9VAEg1pSLFSIZN6S0kiUOzn8SlfIv988UGWnYI6ydpIxmuqvjzn71JMULoVDrizEFpcdePd4UtvVh7vmPF/ptriK6OPSAe0xM7jummdCQZknColXyZlNy8FPbTp4Pmq+fq2u6vEnYIyY7JRqwuVnT7fJJT8YZvL5hrD6+5lgOf3CNU7H9z8eYOtcOyRtzi0M557aYYU5BkSMKqc6vwYuWLEXNgonU6wByDG39k0Pp9gPkCRUGbnY3pB1nEfU3qILdyTi1yXpaP1QX7BLcHSquBOi0Nta++isavV/lNO91w2puFbw1xzwFtow9l992P1r3i596gNiA7Jhs6lQ5/2PAHvHvsXbgZNxwaACoV/rCOQdS+U6LLJxmSkGJKwfrC9Vi8djFanOEVW2nesNHv3y0/Sj83LwWUDraT4HWwcddcIznv2AwyFCVXJjO762ysvXptyIe1FP+a8C/8dQwR+tPXzEa/hH6gZkwAAD9RO6+DNc8RHtaNGjdOlh9s38S++G7ed+iq70rkLhfZDzYhkww7aidPEJyuptWixTrenPomHhrxEGonD8TTi8RvqVQTseealCWstf1u3ndY3Hux6PIM9zD8LKfC+6UYScL1g42ZQeLk6qS23ABeB7ugh7DpthwojQYJIXSwvB/sm1PfxJ9H/rnd2xHDMHAgUlevBAD0S/TPG9AkJyPvh3V+frC8dlSTJm0PFyk/WH3PnshbJ/zCNyVrCrrFdoM6ORmx14Q+D9FTpiD3668ktzcoeRC+m/cd+ib29SY5MSyDYzk0jF++I7n8ixNexDOXPyM5nxC02Yy4G/xLvsZdf7332XMpo3SwCpIk6BPw6exP0SUquID9hUBNh59FfCnAxx8vBr/E46Pw60BFkw52Ua9F+PrKr0Gh476pv1aUDraT4C2iLGukhzb3VZJkgnJruax1rylYg1t/uFV27WIhnt71tNdr0/L11wDafGE9LT46WK5UYvPGjRCiZfsOFC2RLt5/puEMlq5bimJHsVdnJxeb24YVp1fInl+IhgoSF3P8tFNwuptx46uzwm/6f/npL3jj8BtI2nYCj30m/mXJx3B3lgtvY+m6pfj67Neiy9OcDGTEafaCdLBh62C5UonuumAdbEeG7FmXC40rVwpO89XB3r/lfrx37L12b0cM29GjqL7ldgAkH8AXV1UVim5cApOd9U5r2c6XSpSWFkVKB2s/fRpFS4RHVMLRwTZv3izr/jxedxxL1y3FmYYz3vszXh+PtDP1aL1d2u/5sR2PtVuzzFgsaPjoI7/fGj75RInBKojDx9DcFdJJJXzsjtdRSlHeUo69lXuFXThkcrL+JM40kpiiq7wc16y+Bi0VXOKRT+FtXhMrltHnrq2BR4YAv8XZgv1V+9HKtIY9RAzI1wiLwSc5qRvFY0P1duFErmN1x5BvyYehNnRcyeEhL1ViRgb7q/aHfonikm/im9mLUjFKCncNOa++yT38dV1pldZWhkLOw/NQ9SEUNhV2aDti22bOkU4w0JiBtdnQuo/PniX76qkjx4GR4asaKZimJtiPdFwD7K6u8b4kh6LJ0YT9VfvR4myBilLBzbqxv3I/fjq6GmwlebEYGCKT/+ean5HfKO19/WtD6WA7Cd6nMWr8eMl5ed0kn3QkBZ/gYNKY8ML4FzAuY1w7W0lgxg3HqfpTcAwhMSfeog5oS5IQ1fNy2Y5J994D46BBsrZ3/7D7saRPeFIUKY2qFFHxJH7mGiAiNwJweZfLQ64jK4poX5Of/Ru0OTlB03mvVSlfWTFYzk7saC51wYaIc2JyZM/La59VsW3yMV4HOz5D+roOhWnsWNFpqkRil8ZnEU/JnoJbEm+JuExn7TAKhsv8r2teN7qnB4WKsUSLaRhMzqc6QdqvWa6WXQqvq881CxA9cYLfNF85Hy9pCrEiAEDyn/4EfV95nsoPjXgI1/e6HpuKN+H7AjJSkbh8GaJCeFt7WOLde3mXy/HC+Bdg0oibWggRKNvivYgvdRQdbCdBcS4Quu7dJOdNiyLif7kidT4BRktrwy7YIASblw3gEDxZpBOifcyOeSNkrVhRcu6LxjxnjqAJuRBSHVkgepUefeLFFHjyMETHEh1sjvB+qCk1eseH1izHas2oBRA/92rB0QPeKYfXNYcLy2mnZ038g1dTG0nSTGkYmCSsPxVCyxUxoH1euPg4dLhaRz80Guh7Cx9rXx2sm3F7tZWDTYMjbrj+U18a1/QKPB5k2uH+RqQOJi91mkxyPukQxuc8Q1KGREQHy99XMXPmBHWiw1KGeXWwhkES55O/P6+cAzX34iIF/8K+9vxaqLhvtE+zyzFB34ABCC6wArQVBcmKyUJWTHg1zWmzOahDNfTvD1dJx6qFXQyUL9hOgvVwpRIr5JUpA+QPEXsYj7fc4K7yXV77qvZCVRKdHV3bCEC4VKKo6Jvr7FsPHIC7Tp428njdcZyoOyG7fXaPHaUtpdIzhsDBlUp0VIsMdbNuyW1YHWQdjbt3eG38AtsJiA8RS8GXSjQ3uaGiI59QVWGtCGvIla+9y/oMjfIvdx0aIna5vFrXQJz5+XDmk/rAHtYDmqJR2lyKU7ZTkcus5r4Oe5WysJQUBEwj28gsaoUln8hSPNx1zQp4AAdSYCmIjA6W21fbgQNe6zae/MZ8nG04C3d1NZznCyVWw92f+w/A3SDPU/lY7TGcqifHm+9gz21djeJC8SFr/lxVWiuxq3wXXJ7wdLDOYv8Sos7iYiUGqyAO6yJxK+ue3ZLz8skUfLF4KRIMCegV3ws2tw2/3/B7bCwSTkAKRa451ysLovYcAQBofibFzX3NxFlOE9t68KDgemizGZTBgPI/PQTrbvF9NagN6B3fG3paj+f2PudNsJLL5pKO6S6tnLE2ezxYv8rzU+lPgr/nmfPQJaoLCjWNAIDKpb+D4+zZoPn4esliWsje8b1DDiFSXKyzcNv3YesI5XKk5ojseW2Hyby+LxO8DnZPpbChgVzEzCMAwHGWdLDdY7sj1ZSKNQVr8Fr1axHrYOkoE6j0FNy0iYHzR/9zTmm10Pfpg5n7WSSuIbpRG6fLldNBnW0Ivi7a1UajEdpueah5+RVYvvOXwfm+JPEaXTHUcXEATaPs3nthOyDuVWvSmNA7vjcMagOe3v00Xjn4CjFPN6iAmGgsW8vAdFD8y5y/R34s/hG/3/D7oNi2FNZt/ufBul2+b3FnonSwnQSvg42dO1dyXr6WcEaUvKHFxb0X49PZn7a7bQDwzOXP4C8j/wIAUM+biRGpI0BPI0NDqqi24UnaTOI9MVdcIbiemKlTkftl6ExGAOid0Bsr5qxAji7Hq7OTi0FtwA29b5CeMQQJGWRIUzdBOPanptVY1EvY4/TfE/+NPw79I1pmXyZLBysWE18xZwUW9lwoujzDDcWuzKqAxdm+alChCFcHGz1lMgD4DS3yOthw1hMIpdEgbrGwHthXB/vhzA9xU1/pDNhwMY0YgZTPSI3dPgn+Q9XarCzkfvWlvw52wgQAgCb14ulgDQMHIvfLLwWnhaODjZk5U9b9OSBpAFbMWYHeCb2hptRgWAYe1oNTvU0wffCa5PJvTn0TywZJu2oJoehgFX7VxOnj8Pb0t5FukjaUjgS8zu6XxkXVwV4CWcQKv01UNMkifmDYA/joio+kF/gNo3SwnQQ/zNq4apXkvLvKiS2V3Djj20ffxu/W/67dbQOAR7Y/gr/v+TsAoGEF8WJs4srX8eURgbY6rE3fC1eVafr+e5yfL13V52TdSSxcvRCFjkISPw5TB/vRyY7d6HWlRELg2CI8DOxm3Pjs1GeC0/64+Y/494F/w/zN9pA6WD4uua10m+D0hasXhtTz8jrYMSeYC2JIH64OtnnjJgD+Pqi8FCmc9QTCulxo+PhjwWm8DpZlWSxcvRBfnJH++gqXlh07ULWIZLGfqDvpN82Rn4/zV8/ndLAkBtvMDWe7ZNi+RUoH23rgAM7Pny84LRwdrOXbb2Xdnz/X/IyFqxfiZN1Jrw7WrDPDuOsorEvuAICQBSduXnczPuCcd8JF0cEqtBuPnGL/TlKEm394SVHeUt7hTMXzlvMobiY+rp6aWsz8aiasNZxm16fYP58kIVbs311XD1aGT6bNbcPJ+pOwM3bQFB12B+JmOtbheNzk2KqbxdsqFjfKt+SjrKUMmrrQxdL5+KRYsf+T9Se93r+CcMc6pvXS+IJtK/YffOzbm8jF41vvWnDbrAcn6092eDuC27ZYwJaTZDdbgL6aabXBfuJE0PyAgGfpBcTT1ATnuY7rSt01td7rKhRWlxUn60/C5rZ5QzjrC9dj98n1QBOJwQ8KYRl5ou6EpFnGrxFFptNJ8DrY6KlTJOcdlDQIG4o2eGN4UnhYkkWsV+vx+uTXOxzzYSaNRknzPjhGjYfhq01+9mS8NCFqnIjWlrt50559FsZh4n6RviwftDwstw0AGNtFXDcph+ikLnACcA8Vl/tMyhSuIcyTpE+EDUDCKy9C1zXYVzZBT3SSw1PF9YKhYLm4/aFuFDIuQC1iAOgRJ64DDsQ0ZjRshw9DFRfr/U1Dk+t6Srb0dR2KqMmTRadpunTxhhBUtArTsqfBXe6OnEyHyyL+cBKNNC6+6jMRAPD9UArVk4l9oXHECFh37oI6UVrj2iWqC8payiLQSNLGxDvvRMy0qX6Tko3J3s5ML+XBzO1Pl3+9CP2A0HaMPHcPuRsMy+CDEx8gqfwgMgCk/vWvMI0S16J7GKKDnZA5AZnRmWHLzAL9pg3DhsK2Xzwp61JB6WA7CV4HK8cjNdmUDACyL0o3Q2QcGlrjNQroCGx2FwD74EkjySz8ywFAsioBQCNgME4WJjdw9ORJgh6pQgQWWJdCr9KjW6y0njjkOozRcALQdxHeDzWlltR2GmgdHEYjkqfNFJxuVJMXk3BMzX1huWvmmol3IT1W3Bi+vaSZ0iS1vr5o0klMnk/YA+Dt5OQaUwivWANdnvCx5nWw/Be8ilIhKyYLfQ19I2i4Tq7Z/d0o3JQXcDy4F8ZTPU3o2ncQAEDNJTfxzlGh6JfYD1pVBOwGmbb7KrCoSf/E/l4drL53aF9e/mUiavJk4iIkgwFJpCN+59g7oLlB0Fej9mAW3Q1DIHxt83Z16VHpSI8KL5eDNpuDDNb1PXpG5Av+QqMMEXcSvA7WWSwtlq5tJTEuqyvYY1QIhmWgolRwMS5sKtqE4qbi9jcUAFVKhoZVlZzez2cojHWSv11lwm/l/A3csnUbXDINko/XHvfGneVg99hRYCmQnjHUOlrJMJe9XHg/3KxbUiNqd9vAtLaiYv13gvEhm4cMP7fXu5bXweqrm7xfipGkwloRVmiB13D7yrZ4qUw4fr7BK3bBeV74fDrz82E/caLtC5ZSocBSgCOtRyKng+W+Dgfns2goOB0wiUzrftaKxhNE9+muIl+LTKt0KORU/amIxGD50EzL1m1wFvlrRE/UnfDqYO1nJM6ndz1bg3xlxThacxR7KvaQTpOLu9ZsWoeKgqPCbWVZbyWnkqYSbCraFFaddMZigePcOb/fHOfOKTFYhRC4ScdkOyp8UfrCP9h5HaUU2THZ6J/YH3a3HfduuRdbSraE3bze8b3RI5YMF1JHyU2qPktuZMbhU3vWQR6u9pMnIYQmJRmqxESUP/hgSE1elDYKw1KGwUgb8f7x9/HMnvCsrXhDhPZib+JiyOcKRec5UCU8JNUvoR/yzHmoi9fATQONdz8IR0FwB9HsJJ34ucZzQdMAUoEnVKY2r4PN37sBFS3yjdHD4WS98HkUwnGGdD6MtS1OycfOj9ZKX9ehaN0nfq24iotBgcLQlKFIMaVgQ+EGvFXzVsQ6WFV8Auhuubh5EwN2u79HKm0ywTh8OGbvY5G0keiA+Yc/0yxtWF7UVCQ5j6w2xsZCP3AAal56CU0/+HvWVljbrg2pesWatFTQ0dEou/se2I6Ia6BjdDEYljIMUdoovHX0LTy37zl4GA+sZh2o1GQsW8vA+LPwFyULFkNThiLdlI6fyn7CvVvulf2xwNMaoKEP5V17KaF0sJ0EX2LQPEt4ONEXPmYnVybzh4F/wPPjn29/4wA8Nvox3DfsPgCAevZUTMicANUkUsJQFdVWEo7mvGF5T8xAYmbORPYH70tur0dcD7w7411k6bKITCeMJB6D2hBSPyqHuDQSp9aNFa6prKbVuLr71YLT/j7271g2aBmY+TPw7ELxWyrZSIb6x6QLb+PdGe9iXvd5osvzOtivu1TIdlYKh3B1sHwdbXVCvPc3LU2GP2fmSl/XYlAaDWJFfJJ5HWyUNgrvzXgvIqVAA4m6/DIkv/MmAKBnfODQZA9kf/gBp4Pt6Z0fANTJyZLrjpQO1jRiBLI/EM7KDUcHa77qKln3Z9+Evnh3xrvoEdcDKloFhiGZ7OcGJMD45gshl6UpGu/NeC/ktR1y+V+wDlaJwSpIEqeLw38mPYb60g/RvsHN8Ai30MSlAl/o/mJwKWQRK/w24cuwvjLxFTAsg/oC+aMevzWUL9hOgo9bNYj4XvrCl+jjZTNSPL3rady7+d52tw0A7ttyH57Y+QQAeDVolm+IttG3NB4fB7Gs/lZwPQ2ffYaCmbMkt3e89jhmfz0bBfYCYrgepg72vePvyZ5fiLoSUsLOuWmL4HQ34xbV8d2+4XY8t/c5aN75UpYO9sfiHwWnz/56Nj4+Kaz/BACak66MO3Zp6GCb1v0AAH6xO15K1hF9Kutyoe699wSn8TrY6tZqzPl6TrvKgErRtG4dKueQL+hjtf5lLW1HjiB/+gx/HSznhSynrnikdLDNP25G/gzh6mnh6GDrP/gA5+cJj8z4crj6MGZ/PRvHa4+TQhOMG0aNEe7VP8A6fykAQEUJv2C2ulox5+s5WHVuleR2hFB0sArthrFIx214r1O50pUKa4VfHKY9VForvck4HosF4z8fD2sD9yD11c1xSR9Mk7CnpNybwOFxoKipCE7W6fWbvJh4uOLjqlbx5AveTDyQcms5amw1oBpCly/kz59Y/KmoqSh0nJ071kb7pfEFy3AFR1h3cFt47Xa7kdCUOjwOFDYVhh3Lk4OnqQmwCGvPGZstKKmIsZI2sO6Lp4NlrC2yvKSlkFvg3+a2oaipCA6PwzvC9MnJT3Dk3A6vLp7PLg7ExbhQ2FR4wepnX8ooHWwnQWlIrCpmpvBbqC/DUoh+VG4MltfBGtQGvD/j/Q7HqZgZ41Bvr4fjskEA/G25aBORDvF1aQPhsx2z3nkbppHyPFuX9F2CVya9ElYbJ2eJ6yblEJNM6jx7RohrAWfkzAi5jmi1CaxWg/j/vSrow8kX8udrS4cLr4Pd1+PC+cEOSJSnhQQA0zgiAVPHx3l/47Obr8iVvq5DEX2F+LHW5uW1yXRoFa7MuxL3ptwbuVKVXOb7y1fSME+dFjCNXM+fjKdRM5Pcl7wXsjpJWgebZ+6AjZ9AO1Kffiooj8NXImUYOlRiPWRfs957D4Yh8nyKb+1/K14c/yJW56/GGc6IJOPNN2C6TDi3gC8Co6JVmJI9Be/PeB9R2vB0sKbL/S0sTWPadw9dbJQYbCdBqci7Da8lDEW8gSSRGDVGiTkJfEq8mlZjSEr7zL19YdNI8gaTTB6klLrtsqG05IGqThEpgsHdwMZRo0DR8t7nsmOykQ1pfTCPXqXvmO4SgM4QBQcAXaqwjk9NqZEZLaL15VCBhiYuHiljhTt7vYp0kO013eZ1sDdNvB9JYXrmyiHNlBZWAo6GS+qhfPSTvBa1vVpfsmINtJnC55PXwTq5jGE1pUZaVBry9HmCHrztglv3sRwKY7PyAiaR67moqwl9upGiJLwBvK8vrhjd4rqBQceznfl2mEaNCvJZ7h7XHTq1DurkOui6SpxPhgE0mpBFIgLhrxEi0yHn+6+Or7DQY8QIBCd68dndKkqFZGOyN9lPLrTZHKT11ebkwn7i0o/9Kl+wnQTLDas48qX1m1VWMlTLyzyk8DBE1O3yuPBt/rfIb+yYIJsqIJpGVSlpB+P0kelwfwcOm7XNQB4ETWvWimplAzlZdxJrCtZIz8hh99hxiouHtReblQzv2ooLBae7WbekRtTtccFVV4uCLz6AW6B0JD/U395KPhTnw0uVVFyQhKoKa0VY8hpnCamNzdjadLD8l3WHdMkuFxynhW0Dnfn5sB065A0hqGgVTtefxp6WPRGT6fCjLmNOsKg7HSBz4a7nfsdbUH+YSIn461qqvCNAYpmR8YPl7qu13wdpRA9VH/LqYG1Hha0RfVYEuFywfLdGVi1lgORLrDu/Dm7WDZo0A/b1m1CbL7wt/ppQ02oUWArwbf63ssu+AiQGG1ie0n78uBKDVQiBt4OV7vx4w3S5HWz/pP4YmDQQdo8dj2x/BDvKdoTdvMHJg73DhVQ+Sa5SlXA5xH6FJrgO9rzwQ0Obkw1NdhbRwR4+LLo9s86M8RnjEaWKwveF3+PxHY+H1d5w9JtCOK3csS0Vf8iIdeLDU4ajd3xvuHLTUWfwwPHos3AWFgbNx3ewYkUYxmeMR3ZMiC93roM9c2RzWIb04RBOx8ibYDO2to6Ff5h29KVOTFcNAO6qKhjUBozLGIckQxK2lGzBR3UfRayD1aSnQzWgL5ZuZKDe599pqOLiEDVhAubsZZH+E3kJcJVyHaxVOh7c3iIjQW1MTYFpzGjU/PvfaP7R3wu53t72cuc4FfrFU5vbFZqMDJQ/8ADsx8Q74zh9HMZnjIdZZ8bqgtV4atdTYBgGLclRoLvm4I41DAzHCwWXVdNqjMsYh1RTKnaX78Yj2x9Bq0v6ZcQXe0C9AFuItl5KKB1sJ8GXGAysIyoEP8wrd9jtvqH34d6h97a7bQDw4PAHsXzQcgCAdsZkzOo6C7pxRO/nF4PlvGGjAmu2cpjnzEHmG29Ibi8vNg+vTn4VGdoMr9+kXAxqA+Z2myt7fiFiU8iQpH608FCZmlZjTt4cwWmPjX4Mt/a/FfrF1+CNWeK3VKKBDCWOTBPexquTX8XsrrNFl+djsKvTKjvcgQkRtg72Mk7/GR+sg52aLX1di0FpNDBfJdwOXgebGZ2J1ya/hkHJg9q9HTGiJ05E0qv/AoCgEpyGfn2R+eYbsOopdIsjcXZ+eNXXF1eMiOlgx4xBhsh9FY4ONvbqecj4j3S+Q6/4Xnh18qvIi83zJjl5WA+KhmfA8K+nQi6baEjEa5NfE9V/S0GbzYi99lq/3+KuvVbRwSr8OjDrzPjH2EdQf/4i6WA5v0mWZSMXV7sIXEw/2IiVBVRQCBPer/mbud+AZVmUnTnY2U26ZFG+YDsJXgdb/8knkvPypQ7llln73frf4bEdj7W3aQCA5RuX46FtDwEA6t99FwDQ+OWXADgZAwcfB2n86kvB9dS++aYsHezRmqOYtGISztnPeRNl5HYikdDB1paQ+KrrB2GNqptx4+2jbwtOu+n7m/DUrqdg+8fLIXWw/FD/hsINgtMnrZiE9469J7o8zZUknPgzc0GyiMPVwVo4f2C3T41pPrb22Wlh71w5sC4X6v73P8FpvA7255qfMWnFJNHylR2h4bPPUTmZjCT8XOMfg2356SecGTvWTwfL64Fd5dLVtSKlg21ctQpnxwo7WIWjg6155RVZOtj9lfsxacUkHK056vWDpSkaje++B+tc4p2roYVNDM5bzmPSikmiPshSKDpYhXbDyigQzj+05GpDa221suO1YjQ6GmFxksQfprUVwz8ajtZmLrbDJVj4/s36JLr44luUIhQuxoUaWw3crBtqTrB+Mb/SGA85trRT/BizYAV/b3A0oMnZBLY5tM6P7xTtHuFjVWOrgdUdKo5Htq9z4pKodMWfcz6j1ZdwY2zh4vA4UGOruSB6YMZq9eYZBGrPGbsdnppav9/4etys5+KdE6a1FUxTB7XGADwi+vVA+PvTxbi8Q8T/PvBvnCo55J1HzAXL6XGSZT0XTyd8qaB0sJ0Er4M1X3Wl5Lyj0kYBADKiMmStm3fTMaqN+GLOF5idJx7Xk7W+2ZNg99jhGE90f3zc1ffv6BkiWluGBWgaOStXemN2UszrPg8r56yEipY/5CqlUZXCzMVgmcvEZU1zugrHYHk0UAEJcUj4+G3oA+y1ACDZQOQJ7fWu5WOwu3pTF6zQxNAUCd2kD1GTiD+uby1iXgcrFq+WS8yV4svr+vT284Od32M+Hkp7KHJD9NyL3ROLVYibFXDvcC8Tr86mUTeXXM8m7ktSk5Iiuepw7ABDwrUj83//B/Nc/3i1b9zYOHqUxHoYUHo9clauDPJcFePantdixZwVWHVuFSo504mclSsRNWG84Py+Gd9X5F6BL+Z8gWhttKxt8fDXmvff44W3damhxGA7CV4HK0ecHqMjPqp6tV5iTgLvvaiiVUHFytsDm0QeoEw8aQfvZQu0ecOqE0QSPBgGtMEAQ3/5Hq+JhkRvQpAc9Cp9x3SXALR6I+wAtAnC50NNqSUN7ykW0MbFI3mocDKHTkX0onH6OMHpUvA62GUT/4yEXovatY5QpJnSZL/EAW0FJviEPaBNB9terS8AQKOBRkSPrM3LgzYr288PNtGQiAxtRsTi9fwX+bl04KrUzMCJAICqzChkchpZ3nCekuGnmhWTJTqCERaclMjQv39Qsk92TDYoioI6uQ7ajNDnk2UZ0EZjWPdnkjEJScYkMCxDZDoaDe6t/A9uir8JY+KCr32GaytN0YjTx4V9/dNmc1C9AE2XLr+IJCflC7aT4IeT7KeE9X6+lDYTvaHc8nO84brT48Snpz7tsEaUOkNiRqpCEmPyt6sjw9eOc2cFl2VZBozViobPV4hrZQM403AGn536TLZWzu6x4+fa0LZcUtiaSck4W4GwlZybdQfVpQ2EYTxorSjD8XdeEvTWbHGRIeT2WpZRLvIlcKGMpiusFdhfJW4TF4iDk2b56j/5L8vTDdLXtSguF+zHhPW4zvx8tO7a5ecHe7z2OLY1b4tcSIHrECYfZlFz7IDgtGGHmlG3dydpE3dd86UjQ7GnYk+EdLCkHQ0rVwZJmnZX7PbqYFv3S8SoGRYei4Xcn6Xy9Nmn6k95708VaMDlgmHNdjScOy44v1cHS6lxuv40Pj31Kexu+S8ZjMUC26FDfr+1Hj6kxGAVQsB1sC4Z9UQb7OThLzeuNT5jPAYmDYTD48Df9/wdeyvC904cnT4aw1PIkBFVQQylVTVc3VJPW5yS5bSZ7krh/GJ9nz4wDB6MyieeCKldi9PH4YrcKxCjisGBqgN4Zs8zYdWZLWnqgME3AJedHFu2NrhAhHcbIvrVcV3GYXDyYBgGD0ZxlB308/+FsyR4Xv6FodZWGzQNIOUFu8cGl1j0wl0zJ05uw/ay7eLzdYBwimC4q8lLBOtTeITv5DrqV+ssFj+fHosFiYZEzMiZgVh9LLaXbcfK+pUR62B13btBPXo4btnAQH/E/4VLnZaGmFmzcNVuFll7iT7cU1sHwL/ghhhyPZ2l0ObmInrqFNS8+C+0/OR/LfjeN2L6dB5D/37Q9+uLyieegOOUuPY4wZCAK3KvQJw+Dnsq9uCZPc/A5rahOTsBqsH98Yd1DAynhM1IYnQxuCLnCiQaE3Gg6gD+vufvsLmlc098CXw5dxZGxlf3QqN0sJ0EP6wWPXGC5Lx8Ee0Uo3SMBwD+PPLPWNTBIcS7Bt+FW/vfCgDQTZ6AhT0WQn8Zqf9JG9tKNvKa2MBaoTyxc+ci7Zm/SW4v15yL58c9j3RtujeWJjfOaFAbMLNr+/1HAcCcTIYCDSJxKDWtxoxc4TjvA8MfwOLei5Gw9GZ8PFH8lkowJACAaPnK58c9j2k50wSnAQCrJ0OQ61KqOjwqIUS4OljTyBEA4DdUx8dgJ2ROaHc7KI0GMVcI1zLmdbB9Evrgn+P/KVm+sj1ET56MxOfINds1QLNqHDwYXV58AVY9hdzYruS3YSRurU5MkFx3pHSwUePGIf0FYR/WsHSwCxYg7cknJbfXI64Hnh/3PHLNud77kwKF2st7w/Dkn0Iu29XcFc+Pfx494npIbkcI2myGee5c/3bPnfuLGCJWYrAKksRoY/DY6DtRf/bi+cECl0ambDjw8ceLAV9AXUHhYsNf51uu3YJ4fTxKTssPK/zWUL5gOwleB1v3/vuS824q2gQAON8kL3Yz7YtpeGGf8NutXG754Ravp2zt//0XDMug4TOibfRY2mzZPA2NAIDGzz8XXE/Vs8/K0sEerj6MkR+PxBn7GW/2sNwONhI62JpC8kXoXivsL+pm3Hjr6FuC0xauXohHtj+C0rvvCamD5WPpa8+vFZw+8uOR+L+f/090eZorxTf1EHNBJExh62BXkXl9a9jyw+Afnviw3e2Qo4NdV7gOoz4Z1e54dihq33wTleNIVvyhav/Yn2X1dzg9ZKifDtbyHdEDu0pLJdcdKR1s3Xvv4fTQYYLTwtHBVjz5pCwd7N6KvRj58Ugcrj7srYPNsAyqX3gBLVfdCKDNzCKQnWU7MeqTUTheKxyjlULRwSq0H5f0lwif5i73odribOnw15/dbW/LdnS5MfCDgbDZhTRznA7WLbwfjF1eohLDMmh1t3olRsDF9TxlOYcTyhN+x2X32OHwOMDYQ8eVeB2t2H61ultlef6qGeqS+IL1nnMBebBc7+L24vK4YHVZQSHylb4YR9s1G3jPsS5XcFF/LieBZYV10hcC1uny1qbu0Hpk3p8e1hN0fz649UEU17Yl3PVJ6Cu4rJNxXhDf3l8CSgfbSdBcDDZ2wXzJeS9LJ3o7uZZsbtYNFaWCSWPC2qvXYm73ue1uJwB45pG4oGMyqblKR7dp2FTc3zGzRb5SGQYqsxld166VrV2blDUJa+etRVqUfOnNlXnSeuJQxKWRuBg7Xty2a353iXPFsKDzcpC06jPo+/QJmpxqJDKf9sYnWc4Obd8AfVga4XAIp15s9DRyXaiT2iRVfAz26u7SX0WhMIvcF5RGA8OgQX4eo4t6LcIT6U9E1g+WonDP71VInBvQDq7Dfeo6Gg3XElvCqIlEo6lJk75eByUNiowTEpfNnPvNN0Fx1n4J/WBQk2tF8p5jGaiTktB17VoYR8nzWJ2WMw3fzP0G+6v2w2JvBG00ouvataKe0L7evXPy5mDt1WsRo42RtS2e6Bn++Q/8tXepo8RgOwvOG1VOoN6kJYlEWpVwKbJAPIwHNE2DpujIJIHEkpuBiSHt8PN15bxhVeZYwUVZlgjZJX0pfTBpTDBpTNIzcuhVeiTopRNMQqHWkgQijch+qCk14vXxgtO8MAx00bFI7DVQcDLf+YT7cOFhueP+u8vuQ/ygG9q1jlCkmdLC0q+qYsjLla8/MB+fi9XFtr8hGg3U8cLnU5OVBXVqqp/HqFlnRqImMXJ1qxkGUKtRkcDCmODvXcpb2TWlRiMqhfiw0tGk2AqvCQ9FiinFWyGtQ3D7r8vr6nf8+W04GAfUyXVQJ4c+nyzDgjIYwro/o7XR3muZYgFWq8Gtpx7D77W/x7jo4PKNvpKqaG102EUmaLM5yEhBnZj4i0hyUr5gOwl+eE3arxEospA4k9wUf4ZloKbUcHqcePvo25L6TSmoY6ROr/ockU746WC5WLJdLMWfYeFpbkbdO+/CUSDPCq2oqQhvH31bVM4SiN1j73BN2lYLkVq0nhHWb7pZNw5Why5qzrIMLCX5OPDvJ+CqCk4Ha3IRHXN7nXAornxfKCu3jlBhrcDO8p2y53ecJdpnX5s2/mF6vK598TYAgMuF1oPC59OZn4+WLVv8HtqHqw9jg2VD5OLSLAO4XJi9h0HVoV3+07giFJftbkLt9s3eNgH+NbrF2Fa6LSIxWL6jr3//fdh+9teAby3Z6tXBWndInE+GgaehAXXvvCtbp15gKfDmCtAsBdbaiszvDqHltPB16adZrjuOt4++HZZMh7FY0Lpnj99v1j17lBisQgi4G8Q3YUgMvj6t3MILc7vPRd/EvnB4HHjp4Es4WBW+28WkrEm4PJ1IbygLib1STdyD1CeGyBfMYCzCDxfjsGEwjR6N6uefD9kxJBgSML/7fMSqY3Hech4vHXwpLO/MOnud7HmFcLu4l4Zm8ViRr8+mL9Oyp2Fk2khEjR+PfLMdxv+uEDSX54c1+YITgczvPh994oOHlr1w18zx83vw9dmvxefrADW24AIZYvB1bH3j73wn11G9p6dOXI/M2u3INedifvf5MKgN2FOxB982fhuxDlbffwA00yZiyY8Mok/6Jy7puuYi9pprMG8Xi9wj5Fh5+AITLum4c7j6T9E29u4D81VXovqfL8C627/z8a1ZLlTwxBfjyBEwDh2K6uef974wCZFiTMH87vORYEjA2Yaz+N9RkoRm7Z0J9eUjseRHBvpzwkleGVEZmN99PmJ0MThSfQQvHXwprEITAOCu87+/3bXyXr47G2WIuJPgdbBRl0vX5+3LJQ/wtWyleGL0EwDkG7QLcVv/2wAAJ/EM9OPHYmlfHYwOGp5dR0BzsUCgTQdrHCkcu4y9eh4MgwaiZdOmkNvLjsnGk2OexJYtW7zDjOHoYDviPwoA5qQusAAwDBksOF1NqzE5SzjGdOfgO8kfPYBN5f9D7wLhzoEfYuZ1zYE8OebJkG1kuVJ821MtUNcew7zu80LOHy5ppjSMSB0he37j0CGwfP01VGaz9zd+6JDPG2gPlEaD6ClTBKdp8/Kg694dvVOHY3iqvNq54RIzfRo8g3uhcv1mZMdk+00zDh8O4/DhKFv9BbLNOeS3QYPQ+OlnUMVLhBBAdLBnGs50uI3RkybCdNkYWL75NmjalKwpKGwqhDq5HlHjhR13eOIWLoShf3+0bNkScr6usV291+fZhraO2D55BPSTstGyWbzwSf+k/uif1D/k+kNBm82ImemvczfPmoWmtcLZ+JcSyhfsrwyWZSOezRitjcZ9w+5DYjvinO1pS2e46UQClmVBX6DkI19oiv7FaYQjzYW4zn3X3RnbDYdItaE96+GTyT6f/bmsZLZL5Zh1BkoH20l4dbBvCXuM+vJDIfGbzLdIx+5sbhsGfDAA7x57t0Ptu37N9bh94+0AgNrXXkOzsxn1HxJto78OlpRPbPj4Y8H1lD/woCwd7MGqg+j/fn+csp3yZsjKlaJEQgdbXXgCAOBZvV5weigd7JWrrsQDWx9A0eIbsPw98SE5vtTi6vzVgtP7v98frx1+TXR5Xgc7YU/rJeEH2/gF8QB2VQTrYN859k672yFHB/vOsXcw4IMBERty9aXq7896dbCBsf2Gjz7Gqd59/HWw35BjJlQeM5BI6WBr/v0STg8cJDgtHB1s2d13y9LB7irfhf7v98fBqoNtOnXGg4q/POLVwZo0UYLLrjyzEgM+GIDq1mrJ7Qih6GAVLhn4r75IVxUa8+mY9rmAtOPN1TtE/Ev7SruIb+kXUyN8KcJf5/xoR0QJdR4vlS+xS+AL9o5Nd6DGp9PsGSfs3OWb8f1bQ+lgOwmvDvY66ZrB4zJIHCUnJkdyXt+MPZPGhK3XbsU1Pa9pf0MBeBaSL1D7dKKRpGPaZCYq7m/zPOF4IMsy0GRlodvWrYieLBzDDGRQ0iBsWbgFQ5LFvVkDkdSoShCXTurKYopwTWUAWNRT4lwxDDRDByF941oY+gXbf/GWeu2NFzNc7PvYyOSwZExyUdPqsDS6fFzMVwrCS8mu7Xlth9oSd/11gr/TZjOMo0b5eYze2OdG/D3j7xHUwTKgYqLxhztVSF4Y0A6us/jj71RoXkK8YqOnkvMZaKkmxMi0kWHLVARhGVBaLbpt3Yq466/3mzQ0Zag33i+pF2VY6Lp3Q7etW0XriQcyPHU4PrziQzQ4GmB32aBOS0O3rVsRI+IJ7XXTodW4uvvV2HrtVph1ZsF5xQj0zY6Z3TGP64uFkuTUWfA6WJP0g5L3gZUjUPc1oqYpWlq7KQcTebCzRtIOP70h51HqawDgB8OC0mqgSZGXoAUAGpXGWxhfDnqVvt3aUh61hnQMaqPw+VBTaskHI8uy0BpjYM4Q1hTyX1t8EYCw4a6Z6wYvRfyoyOtgkwxJMGvlP/hoI9kPP39grrJSh14ANBrQUcLHmtc/ehgPKFCgKRpGjRHRqujI+cGyDCiVGg3RFPTRsf7TOJmOMyEa+lhyb1Hci0+gHlWIWF1sWF7Hom1kWEClEryvYnWxSDAkQJ0MqMwS9wXDgNJow7o/tSqtt4OkWIBRUVi4+w+4c/CdgomA3qIglAp6tV62rzUPbTaDjvbfD1VMjKKDVRCHlza0HjwkMSdQ0Ej0o2IyEV98jagdHgdeOfhKUD3VcKEOk/ik+iTn/+lTSo61kRiY/aiIHyvDwNNoQc2rr8F+Rl72ZHVrNV45+Ip3v6Wwe+zYVbFLesYQtDSQoa7WE8L6TTfrxp7KPYLTvDAM6gpOYPvTd8NVXh40mffzPV3fPq9UirOFs4kd6w5SYa3A5pLNsue3nyCyK09zW7Y6/zA9XH24/Q1xuWDdvVtwkjM/H80//AAP6/F+se6r3IfvGr6L3LA5w4JxOrDgJwYVe7cGTCNfsJO2WVDzI4nXO06T69rTKC2521i0MTJ+sAwDeDyoefU1tB7wjxNvKt7k1cE2b94ScjUsy8BdV4eaV1+TrVOvaKnAywdfBgDQLMC2WDFo9Rk4Rbyt+Zd+mqJxpOYIXjn4imzrTYDEYK0//eT3W8tPPykxWIUQcLEPVoZ2jn9oyYlJ6tV6LOmzBD3iesDpceJ/R/+HozXC5tWhmJk7E1OyOKmEk7SR8taebYvb8G/0YvsRNXEioiaMR+2rr4bU2SUZk7CkzxLEq+NRb6/H/47+T7a5AYCwdXWBeGNRIWpDO9zCOuSr8q7CuIxxMM+ZjXOJbiR8vMGvAD4PH4sSq9O7pM8SDEwSrgLly8nKn/HGkTck52sPcrXWgM85FwjjORln8I9hwDpCtINlMTh5MG7oQ77iD1YdxA9NP3hrPXcU0+jR0M2ejoXbGcTl++st9X37Iv7mmzFrH4vupwN0wDKy3iOVV2AcMRxx112H2ldfResBcZ07XwhGjOgpU2AaPRq1r74KZ4gONs2UhiV9liDJmIRqWzU2FRPZnWNEX6gnXo6F2xnozgt7APdN6Isb+9wIjUqD47XH8b+j/wvrOgP8X+rl7NelgjJE3EnwZdV4T81Q9IwnyQOJeumhpWhtNB4c/iCAjulg+YfXSTwG42VjsHzQFYh2uuDcdhC0vm2IhzaRoWHDkKGC6+F1sBYu41SMzOhMPDj8QWzZsqVdfrAd8R8FgJiENKKDHSCsUVXTaozNGCs4jffNRR5wpP4L9Dsi/CUTq48FAPRNFC6Kzp83MVgubn8k3YWCDlbnEiJcHaxh4AA0rlzpLZkItIUxwllPIJRGg6hxwvpNrw42Y5w3NyHSxMyYDs+Q3qj8/OugUqOmUSNhGjUSJZ+9jwxumqFfXzRAXtnTGTkzIuLlGz1pEkyXXYZ6ATeusHSw11wDQ79+sKxaFXK+HHOO9/pscrQVlaFmTIDuMi1cX68RXXZk2kiMTBOv8S0FbTYH6aKjp05VdLAKIbhAX7AexgOry9pht5VWV6tXAmFitVg2cBliaZKG75d5yA2Zie0HY7OBsUlLKdyMG1aXlQz9tcOuLtw34qB2cp252H64GTecHuGvslZXK+xuO5jWVmhDHHb+uImdG6vLKroNXzQe6oJkEbe6W9v3Bcu0fbnxX5Ed+YJlXS6wTuHlGVsrWLsddrc9rGHGcGDsdjCt5JoNPM6M0wlPixVgWe807xcsI/0Fa/fYIyItYhyOYFefgG0wra1BX35B67HZwNikvwb5+9PNuL3350sTX8KouMGSjjwOj6ND54qxWoNGNBin+P5fSigdbCfB62DrP5D2zeSHYwqbCiXnLWouwqhPRmF9obCeUy63rb8Nf9zyRwBA3f/+h0prJeo5P1jGp+Yqr4ltXLFCcD2l996LohuXSG7v55qfMeqTUThrP+tNBgpnOO3TU5/KnleI2mISP2LWiccgPzjxgeDvi9YswqM7HkXhoutw1Ufiw2ylLaSU3PfnvxecPuqTUd4SdELwOtghu2v9yuFFCovDgnWF6+TPz1URclW1STX4F4SPTnwkuIxchL7MAMBdXoGWLVvwwv4XMP1L4azVjlL516dQPY9kjB+uOezfrrffxplhw6BigHONJOTRtIZ8vTkFymMGsqVkS1glQMWofu45nJssXO1qe9l2lLWUgWlpQdO3wpprnpJly1F8222S29tXuQ+jPhmFn2t+9hthKv/Tn2C9+S4AQIxGOKHqv0f+izGfyndpCsLtRuPKlX4/Wb74UvQl7FJC6WB/ZTDcW3Sk7cymfjEVjnbqYMPN7gy3VOIlA8uAjZSjSwgoXJgv2F8Svr6kEYdlvRnbwZPIFzpz4U9zSNh23FciKwp7Pfxxv3/r/WhyNIHmjlW3uO6C8/smpP3WUDrYToLm6srGL7lRct5JWcRvMtcsbSnlq4ON0kRh3+J9uL739RJLSaxz8VUAAPssEs/x08FydWhjrxHR2jIstLm56LF/P2JkejimR6Vj7+K9mN1Vvtbtul7Cukm5JGSShwM9Y6LoPEv6hP4SZxkWcZePR+7enTAMDE5W6hJF7M2uyL2iXW1kOCnU+fHdkGyUL6uQS7Q2GtNz5H8V8tpEX4kHr4O9oXfHZETxN90k+Ls6NRVR48f7PbRv6X8LXsx8MXIPcZYBbTZjyX0qpC9e6j+NS+pbfqca1t8R7XX0FeR8art0kVz1+IzxkTl3DAtKr0eP/fsRf7P/sRqTPgZdorqANhql9aIMA12vXuixf79sv+Yccw5em0wqjnk8LmgyMtBj/36YRTyhPUxb2Gdhz4XYt3hfeHaGKlWQ5615/tWADHvAzkbpYDsL7q2RL/ofCj5xRM4DxCvAp1SgKAp6tb7jBs9aciGzWrIevzde7u1VdD8YBpRaDVWUSZZfJkC+YA1qg+yvcL1KD70qPG1d0Da5bdEi+6Gm1NCpdKFXwjBQa3XQx8T5aUO92+BuN74gfthwx31mz6vw3Ljn2reOEERposI6jt7z6fO1x+tg5XoXC6LRgNIJH2vaZAJlMPjFAjW0BlpaGzkdLMOCUtGw6yio9QHHgx8hio6G1sD5I/PHQeCcB6JX6yNTJIRhQNE0VFGmoGvWoDbAqDGCjooCbQh9Plm2ffen1w+WAdwUg5nrrsaGMuHwiu/LkJpWQ6/Wh3Wu6KgoUDr//aB1elk1BDobpYPtJPgEEevevZLznqknOjs5/qi+Q8R2tx3/2PsP7K2Q3kYoqH1Ed6k5eo5swydFnuUSDWwHhaUCLEv8Jqueex72Eydkba/V1Yp/7P0H9lfulzW/3WPHttJtsuYVo7mexMVafxbWmLpZN7aXiTuGAAAYBlWnj2DjgzfCWRps3cVbuJ2ok3ccAvHqYA+Fbz8ohwprBdYXyY/d234m8i/etg5oS+DaV7mv/Q1xudASoHvk4WsR+w4R7yjbgS/qv4igDpYB43Dgxk0elO/wd4FiWQagaczc0IjadST2aj/O1bGWocuMVC1ilmXAMgyqnns+SDPsq4Nt+kHifDIs3DU15P6UqVO3OCz4+56/A+AKTTQ3Y/K3JfCcFF7eN3Fxf+V+/GPvP8LWwTb/6H8emjdtUnSwCiHgv2BV0l+X/BeonPrCScYkLBu4DNkx2XAxLnx88uN2yQKu7n41ZuVyQz5chRpWrfJru+/flEZ4P8xXXoWoyZNR/+67cJwXf7CkmdKwbOAyJKoT4fQ48fHJj3G6QX5Bho5+pXtdcFTix1hsG4t6LsLU7KmIW7wY59NodFm9H+7q4MLm/PkTO4/LBi7D8BRpC7ZTlnN4fMfjkvO1h3COI+W9Hnx+466HjuYASFVFmpQ1CYt7LwYAHKs9hq3NWyOmg42eOhW6ubMwZy+LpBJ/qZtp5EgkLl+O6QdZ9Ch0BbRV+qssUjXCoydORPyNN6D+3Xe9LzpCSH2Vxl49D1FjL0f9u+/CVVwsOl9GdAaWDVyGNFMa7G57W8LltLHQTJ2AOXtZ6IqFk7cu73I5bu57MwDgTMMZfHzy47Cz/gP3Q+7XdmejdLCdBH+BGEX8R33Ji80DAMTp4iTnTTWlYvmg5UE+luGyoMcCzMmbAwAwjRiBB4c9CPPgYQDa4sdAW5k4fX9h/WjsvLmIXSBdJzgtKg3LBy1HoibRz61DDga1AaPTR8uaV4yoOBIXM/QR1qiqabWolu/63tdjes50xC+5EUVD0kS3EaMjsWte1xzI8kHLMSJNXD/K62AL0lU4XidccaojpJnSMClzkuz59X2IObwqOlgHOzhZ+roWg9JoYBo1SnCaNi8P0TNmYGr2VK9WO9LEzJiO2BtI3kJ6lH99YdOoUUi68w5QoLzT9L3I+ZQsSwhgWvY0WTXFpYiePBnxS5cKTpuSNQXdYrtBnZyM6EniOQUAELtgAcxz50puLzM6E8sHLUdaVJrfy5Nh5nRo54V2yxqXMc7rL90eaLMZUWP99bxR48YppRIVQsAN5crRcvFve3KkGQ6PAzWtNXB5pPW1oai316PR3ggAMDiBJX2XIMZDHvB+OlgP6QQZm/B+uOvrZQ3lOD1OVLdWw8W62mQAYehgO1JUAwDcLnKMPWL7wbjR4mwRnFZnq4PFYYG7pgZ6u7gWkn9hEKs6Vd1aLboNAF7ttNbFXJAs4jpbnbecoxx4rSjr8dHBcm3siO6RdbnAWIWPg6euDh5LIxrtjWiwN7R7G6HwNDbCXU/Kkgbqkj3NzXBVV4MFCxc3jdeRsm7pc2JxWGSVPJVso8UCT12d8DacZBvu2lq/4Xsh3PX1fvaTYvD3p9Pj9MroHh7xMLLYOLASJSItDgvqbMJtlQNjsYBp9t8PT3OzMkSsIA4vAG/49DPJebeUbAEAFDUVSc57oOoAJq2chGN1Hav0c+emO/Hw9ocBAPXvvYeCxgI0fEm8Jf10sNzflq++FlxP6Z13oezeP0pu71jtMUxeORn59nzvV1A4Otgvz4auFCVFfRnntbtRPM762Wnhc7X0h6V4evfTOH/ttRi5QvzLstxK6hNvKNogOH3yysl4/4Sw/hMAaO5lrOfOsgti5edknGHVIuYr6fgOh/Mvg2LHSi4Nnwjrmj2NjWjdtRuP7XgMf9jwhw5tQ4yKxx5D3e/uBAAcrfUffq19403kTyOZ1gUWonluXk/8ml0VwfWnA9lVsSuslxgxKp95BoWLhDPn91XuI504w6D5hx9Crqfk939A+Z8ektzegaoDmLxyMo7VHvMbYSq77z7YHngSABCrFR5he27vc1i8drHkNkLBe+7yNK0Ore+9VFA62F8Z/JcNH+vhs4k7ylXfXNW+akkhNIViqCjiBMTIqO16ScHiouhgAflm9L9W3Kzbe41TFOXN0I4ELMv65xn4Twz7er4gMBFqB8O06/4EgH/u/yecLjtoLo+ka2xXwfndrNu7DEVRvylNrFKLuJPg45gJt94iOe+07GnYULQBeeY8yXl9jaijtdE4vORwh9oJAJ6b5wP4Bra5kxD1+uf+OlguDhLoSdnWIAa6rl3RbfOPsrenolU4suSI7Pm1tLbDOtjErJ5oBqCaI6zVpSkaS/sKx7y8MAzSJ12BtL/9TXByRlQGAHhj2+HCcLKEqiuGIC+2AzIYEVKMKWHVjDXPvxq2w4ehSU3x/sbLc27pJ31di6JSIeEW4WOtzc2Frnt3MKzd+yX1+wG/R4/6Hh2Xo/EwLFSxcbj6lgY8POL2gGkMKIrC3Q+avecxZs6VsO7cBW1mpsDK/JmaPRVnG8RNL+S3kQGt06HXyeCM9ImZE1HSXAJVUq1oTWcelmWh790bGWHcn0aNEc9c/gwe2f4IGIaBPiMj5P3NsIz3XF3X67qw71U6OhrmOf73TOx1i9C0Vrgi2qXEJfAq9huFf0OW88XjnVV6Xl8j6ojhbatAOyT2g2WJbyVFURHTKQZCU3SHszP5tom1kYaMbTAMKFp8X3mNKCUj2zQUYzPG4dXJr3ZoHULQFB1W2yiBcx+RfaRpQOxY0zRA0/AwHm8sMOIwDMkmp6ig65qX6VA03ZZ5zs8i4/qmKToi9wHLMqL3Fb8NiiLtDAm3r+Hen/xIGcWycMKNsZ+PFS0B6mE6WMmJO+e+UBE6jhcapYPtJLw62B07JOfldZPVrcHSj0B8/WBtbhse2/EYdpbt7EBLAWon0V1qDnP1en2K9zNWEhdsFdPzMgw8dXWoePwJ2I7Kt817atdTsuvi2j12bCzeKHvdQjTVEqutVhE9r5t148fi0G/5LMui8sQBrL19DpxFwfHyegdJbjlSI//r3BeKi9u37pXwpW0nFdYKfJv/rez5eS9j3yQZfuh6Z3kHrjmXC82bNglO4nWwbrat0MSWki34uPbjiCV+sSwDxm7H7773oGxLwDXIDc1evbYRtatXAQDsnHbaUy+dvBQpHSwYFvB4UPH4E0GaYV8drEWiFjEYBu6qalQ8/gTsJ0/K3vzjO4lMjGJYME3NWLiqHvRx4S9zN+v2ji7sriDxc6vLKntbjMWCpu/9O++mtWuVJCeFEHBvZHS0dGq/SU2GBuVUAOoe1x33D70fiYZEuBk3Vp1bhXON58Ju3vW9r8f87py8JoZz0TEZ/NoOAJSK348owfXEL7kR0dOmoXHFCjhD6Oy6RHXBA8MeQLKGyGW+K/gOx2rkJ2qFVXpNALWGG3I1GkTnMevMgr8v7bsUc7rOQdIdy1GabUTulnNwC2R48udPrJLPA8MewGXpl4k3kjvu+e5K3PJDB4ZgQxCvj5c9Lx3FVTLyqWDEf1VEa6MFl5ELX4JTCEqnw7U9r8XCHgsBEAP73dbdEdPBxs6fD/38KzH1MIu0Kv9s/Ogpk5F0110Yf5RF73Lu2ucrCsnQtBvU4tdXOJivugrxN9+ExhUrYD/pr3P3/VpUJYQ+n/G33IKoCePRuGIFXCHMCrJjsvHAsAe85T55DNfOh2b6JEw9zEJbLlwIZ263ubixDykJW9BYgFXnVslyjfJFHe+fQKWKl3+ddiZKB9tJ8OJ0Q/9+kvPmmHMAyOtEcs25uLnfzYjTS2tmQzG762xMzZ4KAIgaMhRPjn4ScQOI56ufDpYrJafv3UdwPeYrr0T0tKmS20sxpeCmvjchXk1uHBWlkp0pa1AbMDRF2I9WLqbYJLKunr0Ep6tpNYakDBGcNq/7PIzPHI+4665DXb8M0W3wnQ6vaw7kpr43YVDyINHlWU47XdnFGBFP0UDSTem4rEuIDj4AfY8eAEgpOx7+S6VfovR1LQal0cA4VPh8avPyEDVxIq7IvQIzcme0exuhiJk2DTFXkhq+KaYUv2mmUaMQf8NiUKCQYkoFAOi6dQMAP19cMcZnjI+MDnbSRJivvlpw2sTMiV4dbNRloc9n7Ly5iJ4krX1Oj0rHTX1vCjoeMbNnQjMldJx3ctZkXJl3peQ2xKDNZhhH+uuiTaNGKTpYhRBwOlg5GjReUyjHY7PJ2YRCSyFcTMd0sBUtFai0VgIAdC1OzO8xHyZOvsn6+l5yOlhPk/B+OEtL4a6Stueyu+0otBTCwZBhUJqiw9LBdlRb6HKSnXOL7IebcYvqLstayohGsLAQ+hbxTGv+nIhpXQsthaG1ndxx17W6LogOttxaHpZe0dNM9sPrh4q2JDtfU+5wYV0ueBqFj4OrrAzu6mrvMb8QuCoq4CwnkptAzbK7pgbO4mKwYOHgpnlaWrztlqK6tRrlLdJyHsk2VlbCVSa8nmob2Ya7uhrumtDlVZ0lJXDX1Ehur9XVikJLofd46FV63Nz3Zuirm8FUh96G77OkPTAWCzz1/telXH19Z6N0sJ0Er4Nt/EJav/lTGYmxlDSXSM67sWgj5qya0yFhN0CsqJ7c9SQAoOHjj3G05igaVhGtK9PS1kHwOtim1d8Jrqd0+R2o/Nszkts7UXcCc1bNwXkHiU+paXVYnUg4sUMhGsqJppHaKh7fFNPaLtu4DM/vex7nr1mIvG8Oiy7PP2TEtKZzVs3BJ6c+EV2e5mLfGTvzL4gOFgB2lEvnBPDwGkvfhzg/9PfF2S861I7GlcLLs3Y7bAcP4t7N9+Jvu4WztTtKmZZwvgAAME5JREFU+Z//goY/PwEAQRWzav7zKgoXE00nXy6whYsXuyqlO5GD1Qc7ZEbPU/nXp1B6112C036u+RmtbvJS3rJlS8j1lNz2O1Q997zk9o7UHMGcVXO8+SAqWgU340bZvffC8ezLAIAEfYLgso/seAQP//Sw5DZCEZgx3LxOvm9xZ6J0sJ2NWjq7zqshk5GZySeZ8Bmv0drojjmbAGBVKly/9no4IK69pMT2gyWuH5TRKKvuMk+MNqbD7Q4Hir8VOqItZBhQahXsGrLG4G1wdXo7qANkVfKHzy8oqoAsWh8uWIYvh5tp01bqVDoY6MjENgF4taF2DcCqA64HlgFF0XBoADd/Guk2jedFg2EACuS+kqjbHAqWZb33J8JQHlhdVnx08iMwjAdqlQaU0YiceOHQh2/Gt1alRbQ2usOZ9L8UFB1sJ0FzscuEm2+WnHdy9mSsK1wnGrvzhR+iU1EqRGujsfO6jmUQAwCzZB6AVbBfOQFRr3/ul4CiiiOx3thrFwkuy7IsdN26oevq8L4wV8+TX6lFr9J3XAeb3QNNANSzhePFakqNpf0kdLAsi8yJs5HysHBlHF4HO6tr6NqtYvA62JZZl2FocuTLBKaZ0jAiVbwWciCx8+bCduAANKmp3t94S7+b+gr7ucpCo0GCSJ1dbV4edN27w8Pme7OIb+53M3JqcyKng2VZaGLjcM3CUjw8wr9aFMt1vn+6L84bV4yZPQvW7duhkaGDnZ4zHWca5LnWhGwiWNB6A3odPBA0bXLWZBQ3F0OdXIeo8aHjo2BZ6Hr3Cvv+vHPQnXj18KugWBaajIyQy7tZN3QUuS4W9FiABT0WiM4rBG02B+lg466/3ltJ7FJG+YL9leFruH5JwLCg6N/G2yp7kar8DEwaiLemv3XBt3Mpw7BMxJxpAuG1riIbBi6F6zmClZyodhxH7wgKy8LOODDkwyGiYRqGYSKry/8FoXSwnQTLeXu2bN4iOe/RGqIfrbJKJwt5dbC0Cq2uVty35T5sLdna/oYCoLcSjat2P4lH+RoUMFaiZxPV8zIM3DW1KLv/Aa9uUg7/2PsPfHjiQ1nz2j12rDm/Rva6hWiqIRKF1r3CPqZu1o0fCkPXdQXDoOLoHqy+cRKazwVb7dXZSVz8QFXwV4ccKDuJ28vRTreHCmsFvsn/RnpGDt7L2DfZhE/k2lragWvO5QrSPfJ4dbCM2zvsuKFoA96ueTuCfrAsPLZW3P2NB6UbAkZSuA7p+lWNqONqc9sOEO20u1barzlyOlgGcLtRdv8DaP7RX5/tq4MVi2XzsCwDV0UFyu5/ALaj8mVxbxx5g2sHC7bRgmVf26E7mi84r4dtGyL+qfQn3LflvtCmFgEwFktQLWLLqlVKkpNCCLj4ldpneE0MXnKjV+sl5x2ZNhKPjXoMepUeHtaDDUUbZJkEBHJrv1uxuBdJ5mBTEwEATEKsX9uBNrmROjlZcD1Jd9+F6BnT0bRmTchi6FkxWXh89ONI0RAZwM7ynWEVZOCHX9uLRs9pOuNjRecJtC7jWT5oOeZ3n4/Uxx5FTbd4dNtXAWdd8MOWP39iWtPHRz+OCRkTxBvJHfdiXQtmfjVT1JWnI6SbhPdRCHUikTZR2rZYOT9ykmJMEVxGLpouXUSn0WYz7h58N+Z2mwuAaCsPtx6OmA424dZbYFg4F5efYJHR4P+INM+bh6R778XI0yz61pG4ryqRJPf4ytfE6Khemyd+yY2Iv/UWNK1ZA8c5/47NqDZ6/9bm5IRcT/If70P05EloWrMG7irxJK1ccy4eH/04smKy/H5PXL4cmplTcfkJFpoq4Uz+3w34HRb1IiGkkuYSbCjaELbKIbAMpSa7Y3acFwslBttJ8B0T7yUZisxocnHFaKWLUvSM7+n1G7V72v8Anpw9GQBwEkBM/8H457gbkeA+D+v320H7PFAp7qGi695dcD0xM2fCUVAgub1EQyKu6XENtpRvAcDpYMPwgx2QJOxHKxdjTDwsAPR53QSnq2m1qLZzRg6nx1wA2FykGo5QEhJfYILXNQdyTY9rQraR5Uztm7rEoqT5cMQTndJN6RiWOkz2/LquuQAA2tj2QOeHAsU8b+VAaTQw9O8vOI2Pwc7sOrPd65cievJkuKuL0YJnkWRM8ptmGkVqNVOP/QmJ3DQd14n56oHFGJU2KiIa5qjx471KhEDGpI9BYVMh1Mn1MA4PfT7Nc2bLquCUakoVvD5jZkyH5fT+kMvyevr2QpvNMAzx16AbBw+Gu6KiQ+u9GCgdbCfBe2hK6dQA4qcIQJabTZW1CrX2WvSJFy78IJeCxgLvw1Jd34QZuTNQ3fQSrABYj8+DndNABurUeOynT/vZmYnR6mrF+abzsDFEiqKiVLL8bwGig5UzfB4Kl4Ns11UvfD7cjFtUd3mu4Rx0ah0SCy3QNpAhcyEnIN6jl/fZDeR43XEkGZKQbBQeDaA4Hay2yQakRN5Rp9xaHpZG091AEq189Z/8fndEJsa6XHBXC59PZ34+aL0eJ+tOIk4fh1ST9AhQuDjOnYOjnhyHwJJ+zuJiMHY7WLBo5abxQ5V82CcUpc2lXnlPh9pYUCC6vZLmEhRYCuCudopqZXnsJ0/CVS59zlucLShqLkJuTC6MGiNyYnLQM74n7KfPgCkL3dGdaTgDo9qIjOj2jTIxFgtcAZ2pq6JCGSJWEId1ks7S8q109t7uit0AgNKWUsl5vzz7JRZ9J5zRGw6P7ngUz+59FgDQuHIl9lTsQeMaonXl464AMT4GgKbvhXVpJcuWofbV1yS3d6r+FBZ9twhFDjKcraJVYdnVfV/YMWeNhgoSF6N3CNciBsS1tvdtvQ8vH3wZhddcg8QfyPJCba9sJUNwvK45kEXfLcLKMytFt09xOtj4HSdFt9FR9leF/hrxpWUTif25a9s6U14HG04sVwjLN+L3hf34cfxuw+/wzrF3OrQNMcof/jMsz/8bAIK+Nmteegll99wLoE2X3rKVxJtdMgqqdNSnmafy8SdQ+cSTgtNON5z2XhvWnaFVBMW3/Q51//c/ye0drT2KRd8t8h4PNa0GwzIou/tuuN78AACQZEgSXPaeH+/Ba4elnwGhaAmoTd2yWb5vcWeidLCdDGWQjqvyelA5mcG8PpCiKFCgkGpKFa19KxdWr8Nt62+DQ83FuPz0fpx7il5kPxgW0KihSkqUFaPiSTYkw6wVr0cbafhMSkbTzkEdlhwbWqeDJYoW1Df76jbbBznWrI6UTOxsLSwfHhDKEjeoIqhLFcDXoSVKG4U4VcdKg/rBEKeaBlPbseZhGeIVazECNi5SQmm546C6eJmyfMa6KikRtKEDx5phQGk0UCUl+sXSpTjXeA4bijaAZVmoNTqokhKRnSQcJvKwHm/Gt1FjRKop9YJlgF9qKEPEnQSvg43nqsKEYkLmBKwpWINcc67kvB7W/8GzYcGGjjUUAHPdHACrYJ85luhgff1g42IBkALpgrAstDk5yPnoo7C2+Z/J/5E9b0R0sFlEB6u9YrLgdEkdLPfukTl+BpJW3CE4C59AND1nervayJhIrJOZPQkTMmoiXswhXB2sec5stO7ZA3VKW0IT//JwfW8Rf2A5yNLB7vBe54t7L0aXqi4R08GyLAtNXAL+cLcaD4+4LXAiQFN4/K42HWz0jOlo2bIlZGIWT6R0sGBZUHodevwUPBoStg62Rw9kfyQvY59nQY8F2FKyhWiG09PR5YN3RTtND+vxnpu53eZ6k9PkouhgFS4ZPIzn0tKctVNn90uD4r5gJf03I0BXc1f8Z/J/EKuPveDbulS5oNc5w4h7u14q13OE2tFe7bZ3BIFhYGPsGPLREKwuEC4O02E/2F8wl8CV8tuET1BoXi/9hXm4+jAAolOUwvcLttXVits33o5NRf7xi6I6Kx5ddRT9nvgBuQ+vQb8nfsCjq46iqE7Yo5HeSHSX2l1ENuMbg+XrEvNxqEBYloW7uholty9D6z5hjakQLx14Cc/vk66RCpBs6VXnVsletxCNVcRKz7Zrt+B0N+sWfYAAAMV9wVYc3oU1Cy5D6c+7guaptZMEqr2VIt65ElB2khUuVV+2vYStg+Xie24fH1RefrGxqAP+vC5XkO6Rh9fB+l7nawvW4vWq1yOog2XgbrXioZUelKz9ym8SX4TilhWNqPvsMwBA6x5yPuUUzY+UDpZlGbAeD0puX4amdf767HB0sGAYuMrKUHL7MtgOH5a9/a/PfY2q1irindvQiIdWemA8LGyL6TtEvLl4M27feDuanc2yt8VYLGhc6Z+b0LBixS8iyUkZIu4sOJmONld62DfVSDIlozTSMoA5eXMwOHkwAHJh7yjbgTFpY7zTN5+uxvKPDsLlYeBmSK/Q4nDjs70l+PJAGV6/YQgm9kzGHYPu4PxLt4LtmgmgBp6MFODQKVCatrgU/7dWRJeW+sTjYKxWVDz8Z8TMmS3caEcLso+uwj8sdowo/gtw9HmczM5DiylRcn95esT3kD2vELooM1wAkJoIbP47sO8toLUeMMYDw8kwYfdY4RjT/UPvR7TKhPQXJuP42e3o+t99aK0Pftjy+kQxrek/xv4D3WIDZEKOFmDnK8C+tzCqrhH5SEE1CnHjZ+PwwcwPkR0TWT1gOFZqmgwiH/ONAfKdnpxwRih0PcTPpzopCc+O/TO6mrsCIMlGJ+0nI6aDTX7gfjRa6zD0vgNQjfSPbybcfDMYqxUD77kd2kSSI8APDfvKlUTXbUyWdgHyOed+1+CYuwEdeQYk3XU3wHhQ8vs/BElY4nRxaHCQDO9QxxEA0p5+GkxLMyoefQyxC0TCPCA+0/8Y+4+g6y31kUdQ11COoY/uQ0WdsBPV46MfR5opDQB5idtRtiPsDHh9797+/+7Tx2t0fymjdLCdBJ8QoesmXV84LYpcnFId7JqCNXj54MuotFYidX8qfj/g937Ti+qsWP7RQdhcwW/6boaFm/Fg+UcHse7esV5f0JMAzL0H4LXJtyLJfQItq7f6d7BcoouYoD1m6tTQOlhHC/DWFCQ0nMcsrnDCGsqGA60VcNgqMW3lVNwz9N6Q9XsNagN6x/cO+t3veJhScc+Qe0TXY4yKIzpY+ylgxxaAL+LQWgfseBnqjGT0jBHuNMZnjifbs6/Bt0c2414Az+17Dldnqfy2Z9SQB3CXaOFYXVDbuGODhvOA2w6aSyb6Vl+OeocKS7+/GfcPf6DdtY0DSTelC+qJxY6jNkugg+WGbbvGdg25nhhtDCiKgsVhCTo3lEYDfS9hX15tXh7q00z494F/e9vTL6H93rNCRI0fD1d1MZoR7BBjHEZ0pRQoxBvING0mkZ/I6WCHJA8JrYMNOOcAvNcgTnwL3LYR0EUh6vLLRHWwQ1OGenWwhoHi+vA1BWvwcsvL0BVU4HkA+ysPYCKmCM6baEj0np81BW1V0+Y3PIdbE+ZAbCuB187wlOHi+y4CbTZD36/tHK8pWINSQwEGGYBpX0zzu3bCuecvBkoH20nwOlhXufSwL+91anPbBKc7S0pw8tYbkV1ShQcSgOcWqFCBCrz9w9/x4uduZDz3D5zL+Bh1VidW1lejNCoJT466BZWmYHspl4fBWz+dx6KxlDeJhq6sxbiMcaiqJ1Zuvv6fvAZSyPPVWVKC4puXejVs7joBbeTOV4CG82iwuFG2LRl0kwraeArmawCAxQNvlSK97gEcyfwner/9YVBFF2dJCf72RjO61L+N/NzNyHyTlHATOh5P7nwSgHCxfYeNDHU7LVY4dS6UbEuCs1kNbbQbmePq4c4ESou2ACMeDFr2aM1RHDm4DlmPv4u7m7hs4rrGoO053OSBWGvzsXcrKUHJ7cvgPH8erIoG5WGgzc0l+7HrdZR8UAdnUxy00W6cH29DFwBGboS+xl4bcp+CtlFYCG1ODlKfeByVf33K++/MN9+ANjMT5dZyFFoK/ZZdU7AGr615HA983or0OqA8oQQvXfs4MAsYyQ2Jsj4Pen6Y1tf/U+j6rI5r+9qpsPqfG9blgrNU2JrRmZ+PRguNirG0d9ma1rbRgsB95fdNaprv8sVLb/FqQ+11/usuvuVWuMrKoGdZOCzkvuRLJDJ2u9+8Qts613gOrUXnkT9rtt80ANx1UABttAuZY13Q+r5Pu+2k0935CpzdbvRrY+BQ6bnGcyhsKoS72g3HeeHhaP68PvJRKxI56961Bz5B65h+gteSxWHBmYYzKGoqwnN7nwMAJDew+MvrJUhqeh0AoGr0L3/od+3UAoyqBDRTgkmxQM1HC1FZXiF6HnxhLBY4i4q863xy55O4r9qKGBs5/6+teRwZq/8JTWkNtPEsPAtosHFU0HUldm5UN3fAmEICJQbbSfA6WLGaq77sqySxy3KrsCC85PZlUJdUQcUC6XXAQ1+Qh9wfP7ehSy1AMSxcxcWIrquEimWQ0VyNJ3cLawjdDIuvD5Xhmd3P4N8HiRbQsmoVfiz+EY0/EK2rXy1iLgbbvDE45lZy+zK4ysq8RuH1bwtsc99bgNuO8m3xoCxqUCyFtHqyDw994UGXOkDFAuqSKpTcvkxwG13qAJph4Tx/HiW3LxM9HnaPHS8ffFlwvy1cDFZVokbJtng4m9QAS8HZRP4NAOtaiwWXfWTHI0h76l0kWVjQ3Cjl4s1M0PaqbWRocFd5W3y25PZlcBYUkGPkcgMM07Yf//oKTovK2w7tTyR7e/jptqHQUPsUtA2PB87z51F8621+//Y9rj/X+g+7vXzwZdzLda788bz381a8fPBltGzdBgBw17c5+/Ax2LXn2zI8xc6HL4H70SyiqwaAjFp//a9vQZLAffXdt1DTfOdxlZZ6r9mUjUf8p5WUAAwDmgUyj5CXSr42tG8MVmxb5xrP4aEvPEHT+M4VDAunReW95vx31A7sezuojZZVq/xm8y1kYdsvXPeaP69JFniv2Su3O0WvpRN1J3DLD7fg1UOveivEPfSFB8k+yyet9vdS5rfRpRZQAVB7yDWU1gB4SkpDnodArFy29MsHX4bdY8eg8233wL2ft0JdUgWKYZBWx/pdX0L3B/9Cy28/9vU3JLffXpQv2M6Cy1KkY6IlZzWoyRAciYkG4yws9F7k/EMMIP/3fYPi8yJVYJHRIp6QYXX6x0fYaBPu2XwPPtGnkQvGN+uQ34/o4P1wFhb6/VswCaSVfAVQFlVb+3z2gd8vmg1eH78Nfh4wjHceoeMB+H9Z+aJScdpSLQtnpRptR4uCs5ncJhpWPMaXUu8/jf8q8N0ePyLgq0t2FhZ6NbTevFV+PxjKrx3x3Edfa4CMVmyfhLYBhvE+mP22JUKltdLbuQJtx7PSWgnaROrS+noB856ovmU9xa5PoW15aafHaeC++u5bqGl+8/gQZWUEp1EATNwHK83ZCPr6HYfaVnodhKcx/DXUds0FYasLaqOnIXzrQv68+uZKx7dIX0u8YQWAoOWpev8YLL8N/mlBBfwfgOT1J9TuQNLrxO93oWXIvcV4t6+SUSCkvShfsJ0EX3QhbuFCyXnHZowFANGEFm1Ojje9gwFQzo38VsX6z8fP4wGF0ijhqisAYNL639zMgisAAPZpJFlK5dOZqmLJRsxXXinYLi80LZzQZeTe1PVtHZSHIvtQ7jOCzVDCcd6gbeTk+P3Gr4tHrLRefAaJheuy7NBG+75gsNBGu6FmWdxsFU/MqIpve2z4btN3e3yix+SsNq2tb1u9R4DfD7Nvp82iNpb8ta2//20rVS4w8Bj5vSBx2+Lbd1XeVUHr9j1+/L6lmlIRcwWpwaxOaruWtDQpVuBbtzbU+RDcD40GCbfcIjhPZZIau3oFS2jiVHFQ02rB60GoHYHTxNrqyEgUndaSTpKcoiaT86np0pa85pf057Ot6TnTUZukCZrm3xY24Br0wZAQvB9d/ePdk7Mmo3tcd6iTkxF7jbD3Kn9evc8ECihLkL6WfGPSvsszFKDJzfGbN3Abgf/3tl/CkIA2mxF3ww3edQLAuiEUmrjQf7nPx77Q9RW4T9qcnDYZFk3D46PjjjRKB/srIPPNN+Axk7foyjgS4wKAN67mrkCKgiYrCy49ScIojSYxWCHUNIV5g6UF83LbBYoCKKotrhjI8NsAtR7OoSSwyFIsKuLJPvD7wQJwZ6YILs//xgLebWS++QYYA3mBKU9oOx56lR73DLkndKNpFTLH1YPWcHV/Y0gMFgCQPlB0sW9vJtma/AvOcwtUsraX+eYb3kQxDw2wPscq876rQalYACy0MW64xjYFLS93G3ynqs3NRfL993P7SoufF457htyDl641wkWTY1yeALx0rVH6OAZsn9FqvMvz5yPc/QCIExEdoKnUq/SYEzvHuy0Agtdc5ptveB2JxPY78803AI0aoCiUJwD5f7nWbxrFJXRVJtLYc59wURIAyHjtVdF2vHdT2/3le83SXAEXrdnTds35otYDw28l7eCqLkmdPzH48wq0XbNyzuuCHgugV5EiOc8tUIGhyPJlCcDpP/tnIQduw0OTjtiVnuCt/BZu++8Zco93+zwvLyTnhKWAigTK7/oSuq4y33wDNPdhoM3NReNy6SHq9qJ0sJ0EnwHYtEa6Gsn+SlIftqylTHC6NjMTXe65DwDw96UmVMdRSDOl4YYpDwAA6n93Jbqt/wGmxTcCAJZNekAwwQkANCoat431/9Kk1xGNq247V2e3pS2ZgeFqEQvFYLWZmaBNJkRPmQJ1QgKJxwYy5m4gLhcwkzdK69RmOOc2otXMojqWQoNZhZapwzFw/RbBRAhtZiYcauD70VrkrfkO2sxMaDMzET97DtwJMXjoD3rv8XhyzJOiyUCNlSSJwlZhgDZWg9iuraDUDPJm1kAbq4GbovClQzwhzZhJvlgO9NfDYqSQqooN2h4fg91Z3lYfVpuZCV2vnjBdfjnUDFB49bC2/bjqEUTnaaGJJu0YzxkhXH6aAcUCacbUkPvkuw19r16ImjABeWu+g2HwIABA5v/+z7stQFgHO6vrLNwx6ymcyaRxMhN44d5M3DHrKczqOqstBlsXrIP1jcFqMzNhHj8ezpw0PPh7LarjKJi1Zq9sKcmQ5L8fLleQ7tG7rpJqjDzZ9nWXZkrD7K6zsb1lO9yMG9rMTKji4xG76Fq/fePbYejXD6bLLgua5juPJj0dqstGwmKkkL93g98049Ch0A8cgEozi9IfyLGy7uD0wD6mFtosMnyeeNedftv6ofAHHFCXQpOeDvPcuX7XbNyiRYBajbwlCdDGBoSD1Hpyn4y5m8yfmwvT+HFQJyTAdtjf1lGODpY/r24VsKcXBXu0DvflLJW8lsakj8GTY55EmikNNXE0Wo00LCN6wGKkoCnzN8rgt8ECONKVgpoB9v5xMl58sCv0UyZAk5kpeh58YSwWNHCV4GZ1nYUnxzyJGQdZxNjI+V8+668AgKQ774Tzg3/CnkJGFlKMKYL3hzYzk4wcqtXIW/MdPEnio3kdRYnBdhK81EXft6/kvLw20awTr82rTkqCYehQvDr1HjTq3BidPhqWxipsynjam42YmJeNoj79YdCo4GTg1cEC5MtVo6Lx+g1DkJ1gwv3D7ufKm20F26c7gL1wd8uC5uezfjVL+a8vMVlFl3+9CKbVhrJ774X7WoHhcF0UcNtGdHn3DpQf2IlMyoXhbDR+jMrGObUKWSNy/VL0hTidQcGYmeP3myYrC7GDh2NmrhkpxhTcPeTukOvQx8TDCYDqOwi4rC80Z/4PxgQnYEwEht8KFH+KQZy+OJBHRz4KE61DxutzYT2zC6aXPkSPPndhYMCNHa0lQ+u8fpMn5cEHQanVsG7fjm6+WltdFLSTlwL71gOqBlAq0rFk5WZBr2nAvye+hL6J0tcPAOh694I6jtTrpU1RMAwd6lfykqdnXLDN3Kyus/Bdz7cBjwfrF7R1wLru3dG8YQNok49dHfd1GWjtp+2ah2RTFN6YPAcJhgT0jO+J9YXrcf/W+/Hfqf9F9zh/jXGgttMXdUYX/GfSo8gz5yEzJhP/PfJfFDjapGCGAQOgzcwSXFbXq5dk7d60v/4VFlsj+i3bDdYV6zct6c47wLrd6HnbjaDSieuRLo+EFwLt6gxDh0KTmub3W1Z0Foqbi6Hv3z9oaFSTng7jkCHAba+R7Pqf/gUwbsCYQK5BHx1syiN/AUXTKLrhRpguv9xvPammVG/cUS9i+weQ8/rMrW9hWHQ/ZL34BTIMwvcwQOwH/zPpP+hq7oohKUO8nVZL1+2otZQj7oEnUNEYXDxiVtdZ2PlUPoZ7jHD/9UUkGZJwqHorVFlLoQ6jLohx5Ei/de7p8zJiTpRg/YL1YBkGRUM/hyYtHbO6zoKKUuHTU5/ijSlveKVxgZivnAPDIPERqUihdLCdBK+D1WYLPwh84T0pTWrxov3umhrYDhxA95iuUCdwX6ceBr1LgeJ68sXpqiiH9sRRrP3vWLy9owgf7iZfbVE6NeYN7oLbxuYiO4FsY2jKUABEBxvbvS/enb4MKZ6DaMImfx0s19lqMoStqKLGjZP2g9VFwZQ5CdqqvajusgSpf3oB2HI/0HgOtmPHBBOofOlZyqK0r3/mj6u4GPafj+LktIQgyzEhDCYznAB0GdnAxL/AtWI3WuvOAH/i3EM+XCmo7QSAEWlc/d4UgGkWrmYDtCWrpZj8Yz68thIA4gIMuZ0l5bDVMMDwLqg1dgFQAFe/0bC5vwzLTcdx8hQ8yaRDYKwtsB04AE+T/5Bzuild1MdVc74MlMc/kUuTTjoP2sfogdfB8h7G3v0oyIezqBhjujwr2VZKo4Guq/Cx5msRT8icILq87eefoU4TjiU6Tp2S9G01jRoFZ3UxLAg2SDcMGkTaCApmrlSlhttWYMdtO3AApsvG+P3WJ6EPaIqG/ejRoPld5eVoPXSIdKIT/wLsfBUY+Qdg+jPBbRwxQlQH2y+hH6I0UVAn10v6TX+ZVIRUY19IPYXi9fGCxzxq7OVokPCDHbPwbtiOHkWhz2/uomJ4jp2Q2CqBNpuD/KZr0gygz7fF4m0HDsA0ZjQAoLq1GgerD4Y0w9Dl5XlfjC4kSgfbSfCeqs4iYemHL7zOz+oW7yj44dezVSfQ6FBjdPpor15VV9VItlVIOtTseCOentsPH+4uwojceKz4w+ig9R2oOkCSRgCguAzDUoehsnI9abuP/ydf8tFVKmyl17JtG5hWYf2uL01VpG0ui3+VG3dFBeynQhtC69wAXegvYbKfIj60Zxvq0eQIjl0GYrNynrvlZD8cxeVg3W03sJtxo6BR+EVhb8VemGgdso7Xg64Qz87mdcyB3rWt+/eD4rJmGxyN8B2wsp84AVdRMdBQDhOcaASgqqwFgj8+Q2I/ccKbLelpJPvq8ZHXAEQGdrr+tODyqQVNUAf057yG21f/yetgeSs3HtvRY3BXVmJn2U7vF2yuORe39b8NcXp/JxzW5RJ9KXPm54Nx2LGlZIv3CzYQT309HKeFC+rbjhyR7GCtu3fDZmsEADQ6Gv2XP3wYrNsNFiwsnK+vq4J8LTI2n+ucyxIOzJA9UXcCxc3FcJW7g4zOHQX5gM+9BZcVqBEuSmHdu1e07vWxumOotFbCXe2G/ZTw+eTpd8YBa1ToeQCixf+55mcMShrkVwO75aft8FhC+8nuXPEKkjz+X5Luk2fgLhHWOgfCWCxwnD3r95umsAJRNu6Fz3usyTOEt/UMVS3KkZ8PZ3ExoidOlNWG9qLEYDsJby1igdhlIAerSewzVAq9dRuJh60++jn+uovEJFjuwRe7j1ycXvcJH7nJ3vMCyRQAXtz/It448oZ3udX5q2HhPBl9HyR8XWKxWsRl992Pxi+/DLF3hOqThwEA9tLgB6vjROgOFgByj/vn5tuPHvX+XdUqnYbfXENeUNQHyFu1NT/4ZebHkh8Fl/3bnr/ho0PvoHT5cmh3HBLdBl9gIrAWcdU//4kazjP3XKP/g8Tp09Homsj51+4+ivbg4Qp9OAvyAUDwxeV0g/DDVutpk0Lw8LWIPQ2N3t/4GOyGIv8a2+5K0vZHdzyKT099CoCU37tnyD1INASXxAz0//RbV2kZ7vrxLtHzAQC2g+K+vr45BEJUPPEEmj/4GACQ35jvN63m1ddQ9c9/Amjz923dS/SffMEJshHyNhLok1zc3PZC7Tjtf6xbNgrs8znh50PVM39H7X//T3Ca73PC9z4Q4oGvGCRuPR5yHgA4XX8ad/14Fwos/vdn+UMPgfma7KNYCVDz42+gcpV/TN1dEF495tY9/hrbrHyf4WjuecY/3zaXEK9Yu9sOMSzfrkbpXaHDRpFA6WA7Cy5NXJUgICgPgNcUhusjquIKbKs66LrBxpvxl+1/gSOa276Piwn/Fq2KE/fjpLRa6Pv0gSpGvr9rVkxWUFzuQqJSk31zR4nH5+T46qqiolCRYYDOFPyJyeuYfTWivjy3gEbVGPHasbyygDVL16S+GKjM5HxSPh66vA42sMSgEE6PEw32BsEvDUpG2UGeZGMysrWRrclMqVQoSAXYWOFzVZQM1EXzWnb+OAjr1C8YFKnJ6w0JtROnmqyHjpLW5Auh0Rmg79MHXdLFY7i2FDMyXn8Nun590T9RPC78a0PpYDsJXgcbO2+e5Lyj0kcBCI5rScHH/PTqjplfM1dNBQA4JpFYoyq67QFPcw/ZmJkzRZfXZnRB7ldfImrs5aLzBHLPkHvwwvgX2tPcdhHfhWRO6ycJ+2eqKTWu7yXtcZo6YjwmbTyIXqOuCJ7GmTaIxQ8PdKfR2kX8RYXWctKhGZOxoMeCoKHVjiKkgw1F9FRSt1ad6KMV5XSwcjw/t5RswbjPxwW7y2g0iOd0j4Fo8/JgmOZfL3de93l4IO2BiPnBAoAm2oyHl6qRPV/Yr/lfS+MQs5Rk5UdNILWoNWlpgvP6Mj1neoeNEHgorRa5X32J2PlX+/0uRwfrS3kCkPvVlzCNGik5rxCq5CRoP3gFqrHiy7ujdIieNAlTBy/AJ7M+CWv9vjpYHl8d7KWM0sH+SjDPJR21I6rtK5evMBO3kIj+4/mamz5el1dHSPPaUZgB5K22oVuO3++UTofoqVMlly+e7F/sP/qKGYBGg2RDclgm4jyxI4WTti4kg88xMJT7x0Wjxo8n+kidGfXxJIs52ZCMJ0Y/gYxo+W3UpKd7M3P1/cix9k2ukqIiTY+qFK30jCIYR46EmkuyuhjEBBh082hzckJm1kYMbmQn4aYlgpMpgwFRU/x1tHGLBTrzoTdHumVB9Inv06HlW92tmPHlDKwvWi86j9rqQPOWLd565FETJ0rGwkOR4/uSwj3P+Ofbgu7kpUIsg/hionSwnQSfAWhZJe2/ubuceJSWNgsnEgEAZSCZnKxPkZtWrmboCSuJm3jdPrgLMs6oQZRe+q2fXk1iQ7otpCayp9lHB8tlojatE6+p7Cwrw/n5C9Dy03bxjXDZyAyX7PPi/hdxz4/3QBUfL5lF7FADP1v9Y5eqqCio4+Jg1pu98phQNFQUAgDsW0gsmzYaQKnbgo5u1o3PTn8muZ6KvVuxcfJgnNwdfDz4WPC20m2Cy/55JYPUHf7JOXR0NFRxsYAxHm43OcdN338PlmWJWbZMVLGxXlkOpdVw//fvMEP5wbqitHCa/EMUzZtIDNRd2xb/djIkt+Db/G/9tx8T7a36JYnLhYZPhL9ynPn5sK33j0uuPLMSz5Y/6x1qVsXHg44SHs5Xmc2C8qSgJjRb8Oy7bhR/JdyOP77biKb3PgTQlv/Ax5kBeO+xwKFu3g9WHRcHVcCQLG0yAb7DzBoT/Cv++8M6nTg/fwEav/ra7/ew/GABxFZZcX7+Alj3tM+nmK2uxbPvumHaK54rYai0oPT2Zfhpw7uY/+18eIy6kGElX3x1sDy9Nhcghk8F4cu1cseaH7mjI2BI31GULOJOgtKQh5txuLR9Ey+diNeLx2vVSUkwDBuKxf2W4Cpu6IShgROZgDaGq6Wblg7DsKHeZYZkxSErXvgt75FRj0BNqcFgK9hhAwDsgKtfN2hOFIDWtz1o+co2xsHCusXMN9+Ap7kZpcuWw9Mk7BcJAJmpPVGRoUEiNwxe3lKOwqZCGAT0goGczqBgzvGP12qzs2EYOAC942OQYpQuhWaKS4YdANWLrEeTmQljln925GXplwku+8xlz0APLdLfX4yTRzejy+bdcFqDM5d5yUfvBP+v7dTHHwelVuP89u1BOlRtbg5J4ujSBfZKAChEXfdEzPlgAD6Z+Qn6J8n7GtP16e2vgx0mrIMdkChsPObK7QJ4/GUPhgH90bJ5s1/IgK+3PDzV/7rW5uWBNkXhXxOuDann5jGFCCdo8/LwzvQnvCGTels9yl1t58owcCC0WcIxWV3vXqD1occWuzz/PBqtdci7ZQ8c8L92kh98AKzbjZzFC+DuSpJ69H37omnt995wibcdw4ZCk+af+NMtthvONZ6DfsCAYB1sly5EB8uTNRKI85+HJ+1vTwOgUHjNNf7JVSC6eb7gv+/9LsRXy/thlKkv7P/4HEyzeLZ9n4Q+eGf6O0F5ERmv/gc1dcXIu2sXKppbBZd1/PvPyLCpYD/yN1hdVpxpOANVzmXQ+H4NSGAaO9bv35aBuTAfaQst+B7rZFMyhiQP8WqyhYhdMD9IQnUhoMJ5C/4tQ1FUDYCiCKwqEUCt5Fy/XZTjI45ybMRRjk1olOMjTkePTTbLsoLloJQO9iJDUdR+lmXlB79+YyjHRxzl2IijHJvQKMdHnAt5bDp/kFpBQUFBQeFXiNLBKigoKCgoXACUDvbiI1x+RYFHOT7iKMdGHOXYhEY5PuJcsGOjxGAVFBQUFBQuAMoXrIKCgoKCwgVA6WAvIhRFzaAo6jRFUecoinq4s9tzqUBRVCZFUZspijpBUdRxiqLu6ew2XWpQFKWiKOoQRVHfdXZbLjUoioqlKOoLiqJOURR1kqKoYHuo3ygURf2Ru6eOURT1KUVReumlfr1QFPUORVHVFEUd8/ktnqKoDRRFneX+H7EapEoHe5GgKEoF4DUAVwDoA+A6iqI6VqPs14MbwP0sy/YBMArAHcqxCeIeEHtehWBeBrCOZdleAAZCOU4AAIqiugC4G8AwlmX7AVABWNS5rep03gMwI+C3hwFsYlm2O4BN3L8jgtLBXjxGADjHsmwBy7JOAJ8BkF9Z/VcMy7IVLMse5P5uBnlAXhpFki8BKIrKADALwFud3ZZLDYqizADGAXgbAFiWdbIs29ipjbq0UAMwUBSlBmAEENq89VcOy7LbAAR6dF4F4H3u7/cBzI3U9pQO9uLRBYCvw3AplE4kCIqicgAMBrBHYtbfEi8B+BMARmK+3yK5AGoAvMsNob9FUZS0r+BvAJZlywC8AKAYQAUAC8uy4hX5f7uksCxbwf1dCUC6tqpMlA5W4ZKBoqgoAF8CuJdlWfHCqL8hKIqaDaCaZdkDnd2WSxQ1gCEA3mBZdjAAKyI4xPdLhoslXgXyEpIOwERRlLAPoAIAgCWymohJa5QO9uJRBsDX0DWD+00BAEVRGpDO9WOWZb/q7PZcQlwG4EqKogpBwgqTKIr6KPQivylKAZSyLMuPeHwB0uEqAFMAnGdZtoZlWReArwBc+Ar3vzyqKIpKAwDu/9WRWrHSwV489gHoTlFULkVRWpBkg28llvlNQFEUBRJDO8my7L86uz2XEizL/pll2QyWZXNArpkfWZZVvkI4WJatBFBCURRvQzQZwIlObNKlRDGAURRFGbl7bDKUBDAhvgXAmWXjJgDSHqIyUezqLhIsy7opiroTwA8g2XzvsCx7vJObdalwGYAbARylKOow99tfWJZd23lNUvgFcReAj7kX1wIASzu5PZcELMvuoSjqCwAHQTL1D+E3XtGJoqhPAUwAkEhRVCmAJwD8A8AKiqJuBXFMWxix7SmVnBQUFBQUFCKPMkSsoKCgoKBwAVA6WAUFBQUFhQuA0sEqKCgoKChcAJQOVkFBQUFB4QKgdLAKCgoKCgoXAKWDVVD4jUNRVAJFUYe5/yopiirj/m6hKOr1zm6fgsIvFUWmo6Cg4IWiqCcBtLAs+0Jnt0VB4ZeO8gWroKAgCEVRE3j/WYqinqQo6n2Kon6iKKqIoqirKYp6nqKooxRFreNKXYKiqKEURW2lKOoARVE/8CXoFBR+iygdrIKCglzyAEwCcCWAjwBsZlm2PwAbgFlcJ/sfAAtYlh0K4B0Az3RWYxUUOhulVKKCgoJcvmdZ1kVR1FGQcp/ruN+PAsgB0BNAPwAbSOlbqEBs0hQUfpMoHayCgoJcHADAsixDUZSLbUvgYECeJRSA4yzLju6sBiooXEooQ8QKCgqR4jSAJIqiRgPEgpCiqL6d3CYFhU5D6WAVFBQiAsuyTgALADxHUdQRAIeh+I8q/IZRZDoKCgoKCgoXAOULVkFBQUFB4QKgdLAKCgoKCgoXAKWDVVBQUFBQuAAoHayCgoKCgsIFQOlgFRQUFBQULgBKB6ugoKCgoHABUDpYBQUFBQWFC4DSwSooKCgoKFwA/h8jqETRevIM/gAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 504x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def TogglingFrameH(params, **kwargs):\n",
    "    N, TFH = params['N'], []\n",
    "    Hk, matrx = np.zeros((2**N,2**N)), np.zeros((2, 2))\n",
    "    pulses, opH = params['pulses'], params['opH']\n",
    "    for p in pulses:\n",
    "        for op in opH:\n",
    "            for i in range(N):\n",
    "                matrx = sparseMatrices(p@op@(np.linalg.inv(p)))\n",
    "                Hk += R[i]*tensorOperators(matrx, a = i, b = N-1-i)\n",
    "        TFH.append(Hk)\n",
    "        Hk, matrx = np.zeros((2**N,2**N)), np.zeros((2, 2))\n",
    "    return TFH\n",
    "\n",
    "TFH = TogglingFrameH(params)\n",
    "\n",
    "def TimeEvolOpForTFH(params, **kwargs):\n",
    "    TFH, unitary_timeOp, expTFH, tau, n = kwargs['TFH'], [], np.eye(2**params['N']), params['tau'], params['n']\n",
    "    for i, hk in enumerate(TFH):\n",
    "        expTFH = expm(-1j*tau*hk/n) @ expTFH\n",
    "    t_list = np.arange(0, 10, tau)\n",
    "    unitary_timeOp = [np.linalg.matrix_power(expTFH, i) for i, t in enumerate(t_list)]\n",
    "    return unitary_timeOp, t_list\n",
    "unitary_timeOp, t_list = TimeEvolOpForTFH(params, TFH = TogglingFrameH(params))\n",
    "\n",
    "params['N'] = 1\n",
    "params['n'] = 4\n",
    "params['pulses'] = [I, X, Y, Z]\n",
    "params['opH'] = [X, Y, Z]\n",
    "params['alpha'] = 5\n",
    "n, N, r, op, pulses, psi_nm, R, alpha = initialVals(params)\n",
    "\n",
    "mss=10\n",
    "\n",
    "print(params['tau_list'])\n",
    "\n",
    "plt.figure(figsize=[7,5])\n",
    "for tau in params['tau_list']:\n",
    "    params['tau'] = tau\n",
    "    F = []\n",
    "    uOp, t = TimeEvolOpForTFH(params, TFH = TogglingFrameH(params))   \n",
    "    psi_t = [normalizeWF(u@psi_nm) for i,u in enumerate(uOp)]\n",
    "    F = [1-np.power(np.abs(np.vdot(psi_nm, pt)), 2) for pt in psi_t]\n",
    "    plt.plot(t, F, \"--o\", label = f\"N={params['N']}, τ={tau}\", ms=mss)\n",
    "    mss -=2\n",
    "    plt.yscale(\"log\")\n",
    "    plt.legend()\n",
    "    plt.xlabel(\"Time\")\n",
    "    plt.ylabel(\"Fidelity\")\n",
    "    plt.title(\"Mitigating the noise with Pulse Sequences different τ\")\n",
    "    plt.grid('on')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "0bb44e84",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 504x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def H_noise(params, **kwargs):\n",
    "    N = params['N']\n",
    "    op = params['opH']\n",
    "    Hnoise = np.zeros((2**N, 2**N))\n",
    "    for i in range(N):\n",
    "        Hnoise += tensorOperators(sparseMatrices(op[0]), a = i, b = N-1-i)\n",
    "    return Hnoise\n",
    "# print(H_noise(params))    \n",
    "\n",
    "def utimeOpH(params, **kwargs):\n",
    "    H = kwargs['H']\n",
    "    t_list = np.arange(0,1, params['tau'])\n",
    "    unitary_timeOp = [expm(-1j*t*H) for t in t_list]\n",
    "    return unitary_timeOp, t_list\n",
    "\n",
    "def TogglingFrameH(params, **kwargs):\n",
    "    N, TFH = params['N'], []\n",
    "    Hk, matrx = np.zeros((2**N,2**N)), np.zeros((2, 2))\n",
    "    pulses, opH = params['pulses'], params['opH']\n",
    "    for p in pulses:\n",
    "        for op in opH:\n",
    "            for i in range(N):\n",
    "                matrx = sparseMatrices(p@op@(np.linalg.inv(p)))\n",
    "                Hk += R[i]*tensorOperators(matrx, a = i, b = N-1-i)\n",
    "        TFH.append(Hk)\n",
    "        Hk, matrx = np.zeros((2**N,2**N)), np.zeros((2, 2))\n",
    "    return TFH\n",
    "# TFH = TogglingFrameH(params)\n",
    "# print(TFH)\n",
    "\n",
    "def avgHFromTogglingFrameH(params, **kwargs):\n",
    "    n = params['n']\n",
    "    N = params['N']\n",
    "    avgH = np.zeros((2**N, 2**N))\n",
    "    TFH  = kwargs['TFH']\n",
    "    avgH = sum(TFH)/len(TFH)\n",
    "    return avgH\n",
    "# print(avgHFromTogglingFrameH(params))\n",
    "\n",
    "def TimeEvolOpForTFH(params, **kwargs):\n",
    "    TFH, unitary_timeOp, expTFH, tau, n, T = kwargs['TFH'], [], np.eye(2**params['N']), params['tau'], params['n'], params['T']\n",
    "    for i, hk in enumerate(TFH):\n",
    "        expTFH = expm(-1j*tau*hk/n) @ expTFH\n",
    "    t_list = np.arange(0, 10, tau)\n",
    "    unitary_timeOp = [np.linalg.matrix_power(expTFH, i) for i, t in enumerate(t_list)]\n",
    "    return unitary_timeOp, t_list\n",
    "unitary_timeOp, t_list = TimeEvolOpForTFH(params, TFH = TogglingFrameH(params))\n",
    "    \n",
    "def F_tvals(params, **kwargs):\n",
    "    H, Utop_present = kwargs['H'], kwargs['Utop_present']\n",
    "    H_present = kwargs['H_present']\n",
    "    F_t, Ft2, T_list, UToP = [], [], [], []\n",
    "    for i in params['tau_list']:\n",
    "        params['tau'] = i\n",
    "        if H_present == 'True':\n",
    "            unitary_timeOp, t_list = utimeOpH(params, H = H)\n",
    "        elif Utop_present == 'True':\n",
    "            unitary_timeOp, t_list = TFHutimeOp(params, H = H)\n",
    "        UToP.append(unitary_timeOp)\n",
    "        T_list.append(t_list)\n",
    "        psi_t = [normalizeWF(np.matmul(unitary_timeOp[i],psi_nm)) for i in range(len(unitary_timeOp))]\n",
    "        F_t.append([np.power(np.vdot(psi_nm, pt), 2) for pt in psi_t])\n",
    "#         Ft2 = [1-f for i in range(len(F_t)) for f in F_t[i]]\n",
    "#         t_list = [(i**2)*j**2 for j in t_list]\n",
    "#         plt.figure(figsize=[7,5])\n",
    "#         plt.plot( t_list, Ft2, label = f\"N={params['N']}, τ={params['tau']}\")\n",
    "#         plt.xlabel(\"$\\mathregular{(τT)^2}$\")\n",
    "#         plt.ylabel(\"$\\mathregular{(1 - F)}$\")\n",
    "#         plt.grid('on')\n",
    "#         plt.legend()\n",
    "        Ft2 = []\n",
    "    plt.show()\n",
    "    return unitary_timeOp, psi_t, F_t, Ft2, T_list, UToP\n",
    "\n",
    "def plottingFidelityVsTaus(params, **kwargs):\n",
    "    Utop_present = kwargs['Utop_present']\n",
    "    H_present = kwargs['H_present']\n",
    "    unitary_timeOp, psi_t, F_t, Ft2, T_list, UToP = F_tvals(params, H = H, H_present = H_present, Utop_present = Utop_present)\n",
    "    plt.figure(figsize=[7,5])\n",
    "    plt.xlabel(\"Time\")\n",
    "    plt.ylabel(\"Fidelity\")\n",
    "    plt.title(\"Mitigating the noise with Pulse Sequences different τ\")\n",
    "    plt.grid('on')\n",
    "    for i in range(len(F_t)):\n",
    "        plt.plot( T_list[i], F_t[i], label = f\"N={params['N']}, τ={params['tau_list'][i]}\")\n",
    "        plt.legend()\n",
    "    plt.show()\n",
    "    pass\n",
    "\n",
    "params['opH'] = [X, Y, Z]\n",
    "params['pulses'] = [I, X, Y, Z]\n",
    "params['n'] = 4\n",
    "# H = avgHFromTogglingFrameH(params, TFH = TogglingFrameH(params))\n",
    "# plottingFidelityVsTaus(params, H = H, H_present = 'True', Utop_present = 'False')\n",
    "# print(utimeOp(params, H = H, H_present = 'True', Utop_present = 'False'))\n",
    "# print(H)\n",
    "\n",
    "H = utopFromTFH(params, TFH = TogglingFrameH(params))\n",
    "plottingFidelityVsTaus(params, H = H, H_present = 'False', Utop_present = 'True')\n",
    "# print(TFHutimeOp(params, H = H, H_present = 'True', Utop_present = 'False'))\n",
    "# print(H)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "023d749b",
   "metadata": {},
   "outputs": [],
   "source": [
    "params['N'] = 5\n",
    "params['opH'] = [X]\n",
    "params['pulses'] = [I, Z]\n",
    "params['n'] = 2\n",
    "n, N, r, op, pulses, psi_nm, R, alpha = initialVals(params)\n",
    "def TogglingFrame_Ising(params, **kwargs):\n",
    "    N = params['N']\n",
    "    TFH = []\n",
    "    pulses = params['pulses']\n",
    "    for p in pulses:\n",
    "        Hk, matrx1, matrx2 = np.zeros((2**N, 2**N), dtype=complex), np.zeros((2, 2)), np.zeros((2, 2)) \n",
    "        for op in params['opH']:\n",
    "            for i in range(N-1):\n",
    "                matrx1 = sparseMatrices(op)\n",
    "                matrx2 = sparseMatrices(op)\n",
    "                if N%2 == 0:\n",
    "                    if i%2 == 0 and (i+1)%2 != 0:\n",
    "                        matrx1 = sparseMatrices(p@op@(np.linalg.inv(p)))\n",
    "                    elif (i+1)%2 == 0 and (i%2!=0):\n",
    "                        matrx2 = sparseMatrices(p@op@(np.linalg.inv(p)))\n",
    "                elif N%2!=0:\n",
    "                    if i%2 == 0 and (i+1)%2 != 0:\n",
    "                        matrx2 = sparseMatrices(p@op@(np.linalg.inv(p)))\n",
    "                    elif (i+1)%2 == 0 and (i%2!=0):\n",
    "                        matrx1 = sparseMatrices(p@op@(np.linalg.inv(p)))\n",
    "                Hk += reduce(sp.kron, (sp.eye(2**i), matrx1, matrx2, sp.eye(2**(N-2-i))))\n",
    "        TFH.append(Hk)\n",
    "    return TFH\n",
    "# print(TogglingFrame_Ising(params))\n",
    "\n",
    "# H = avgHFromTogglingFrameH(params, TFH = TogglingFrame_Ising(params))\n",
    "# plottingFidelityVsTaus(params, H = H, H_present = 'True', Utop_present = 'False')\n",
    "\n",
    "# H = utopFromTFH(params, TFH = TogglingFrame_Ising(params))\n",
    "# plottingFidelityVsTaus(params, H = H, H_present = 'False', Utop_present = 'True')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "9e08b0d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0. 0. 0. 1.]\n",
      " [0. 0. 1. 0.]\n",
      " [0. 1. 0. 0.]\n",
      " [1. 0. 0. 0.]]\n",
      "[[ 0.+0.j  0.+0.j  0.+0.j -1.+0.j]\n",
      " [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]\n",
      " [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]\n",
      " [-1.+0.j  0.+0.j  0.+0.j  0.+0.j]]\n",
      "[[0.+0.j 0.+0.j 0.+0.j 0.+0.j]\n",
      " [0.+0.j 0.+0.j 2.+0.j 0.+0.j]\n",
      " [0.+0.j 2.+0.j 0.+0.j 0.+0.j]\n",
      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]\n"
     ]
    }
   ],
   "source": [
    "lst = [np.eye(1), X, X, np.eye(1)]\n",
    "XX = reduce(np.kron, lst)\n",
    "print(XX)\n",
    "\n",
    "lst = [np.eye(1), Y, Y, np.eye(1)]\n",
    "YY = reduce(np.kron, lst)\n",
    "print(YY)\n",
    "\n",
    "print(XX+YY)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5eb15740",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Using Walsh Indices\n",
    "\n",
    "# Generates WI and stores as a col vector in dictionary corresponding to the site [key]\n",
    "def WalshIndicesGenerate(params, **kwargs):\n",
    "    N = params['N']\n",
    "    n = params['n']\n",
    "    qbit_wi = {}\n",
    "    for i in range(N):\n",
    "        qbit_wi[i] = np.full(n, i)\n",
    "    return qbit_wi\n",
    "# print(WalshIndicesGenerate(params))\n",
    "# qbit_wi = WalshIndicesGenerate(params)\n",
    "\n",
    "### Functions below are useful to construct the avg. H from WI\n",
    "\n",
    "# Generates WI to decouple within a given cutoff\n",
    "def WI_Decouple_cutoff(params, **kwargs):\n",
    "    N, n, cutoff_dist = params['N'], params['n'], kwargs['cutoff_dist']\n",
    "    WIR_x, WIR_y, wirx, wiry = [], [], [], []\n",
    "    for i in range(0, N, 1):\n",
    "        if i%cutoff_dist <= cutoff_dist-1: \n",
    "            wirx.append(i%(cutoff_dist))\n",
    "            wiry.append(i%(cutoff_dist))\n",
    "            if i%(cutoff_dist)+1 == cutoff_dist:\n",
    "                WIR_x.append(wirx)\n",
    "                WIR_y.append(wiry)\n",
    "                wirx, wiry = [], []\n",
    "    return (WIR_x, WIR_y)\n",
    "\n",
    "# print(WI_Decouple_cutoff(params, cutoff_dist = 3))\n",
    "# WIR = WI_Decouple_cutoff(params, cutoff_dist = 3)\n",
    "\n",
    "# Generates the terms in H for WI with cutoff\n",
    "def HamiltonianTermFromWI_cutoff(params, **kwargs):\n",
    "    N, matrx, WIR = params['N'], kwargs['matrx'], kwargs['WIR']\n",
    "    lst, Lfinal = [I]*N, []\n",
    "    for wir in WIR:\n",
    "        for j in range(len(wir)):\n",
    "            for k in range(j+1, len(wir), 1):\n",
    "                if wir[k]==wir[j] and k!=j:\n",
    "                    lst[k] = matrx\n",
    "                    lst[j] = matrx\n",
    "                    Lfinal.append(lst)\n",
    "                    lst = [I]*N                   \n",
    "    return Lfinal\n",
    "# print(HamiltonianTermFromWI_cutoff(params, cutoff_dist = 4, matrx = X, WIR = [[0, 0, 2, 0], [0, 1, 0]]))\n",
    "\n",
    "# This function gives the final H given a walsh_seq with a cutoff dist. \n",
    "def WI_HamiltonianFinal(params, **kwargs):\n",
    "    N = params['N']\n",
    "    H, cutoff_dist, matrxs, lst = np.zeros((2**N, 2**N)), kwargs['cutoff_dist'], kwargs['matrxs'], []\n",
    "    WIR = WI_Decouple_cutoff(params, cutoff_dist = cutoff_dist)\n",
    "    for matrx in matrxs:\n",
    "        for w in WIR:\n",
    "            lst = HamiltonianTermFromWI_cutoff(params, cutoff_dist = cutoff_dist, matrx = matrx, WIR = w)\n",
    "            for l in lst:\n",
    "                H += reduce(sp.kron, l)      \n",
    "    return H\n",
    "# print(WI_HamiltonianFinal(params, cutoff_dist = 3, matrxs = [X, Y]))\n",
    "\n",
    "# WI_Sequence(params, WIR = WI_Decouple_cutoff(params, cutoff_dist = 3))\n",
    "# WI_seq, c = {}, 0\n",
    "# for i in WIR_x:\n",
    "#     WI_seq[c] = qbit_wi[i]\n",
    "#     c+=1\n",
    "# print(WI_seq)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "a1574f0c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1  1  1  1]\n",
      " [ 1 -1  1 -1]\n",
      " [ 1  1 -1 -1]\n",
      " [ 1 -1 -1  1]]\n",
      "[ 1 -1  1 -1]\n",
      "[ 1  1 -1 -1]\n",
      "[(1, 1), (-1, 1), (1, -1), (-1, -1)]\n",
      "[array([[1, 0],\n",
      "       [0, 1]]), array([[ 0.+0.j, -0.-1.j],\n",
      "       [ 0.+1.j,  0.+0.j]]), array([[0, 1],\n",
      "       [1, 0]]), array([[ 1,  0],\n",
      "       [ 0, -1]])]\n"
     ]
    }
   ],
   "source": [
    "H = np.array([[1, 1], [1, -1]])\n",
    "\n",
    "def WF_Conditions(tupleprdt, **kwargs): # tupleprdt is a list\n",
    "    for i, tprdt in enumerate(tupleprdt):\n",
    "        if tprdt[0] == tprdt[1] == 1:\n",
    "            tupleprdt[i] = I\n",
    "        elif tprdt[0] == -tprdt[1] == 1:\n",
    "            tupleprdt[i] = X\n",
    "        elif -tprdt[0] == tprdt[1] == 1:\n",
    "            tupleprdt[i] = Y\n",
    "        elif tprdt[0] == tprdt[1] == -1:\n",
    "            tupleprdt[i] = Z\n",
    "    return tupleprdt   \n",
    "print(WF_Conditions(tupleprdt = [(1,1), (1,-1)]))\n",
    "\n",
    "def WF_Generate(params, **kwargs):\n",
    "    N, lst, W_x, W_y, tupleprdt = params['N'], [H], kwargs['W_x'], kwargs['W_y'], []\n",
    "    power = max(W_x, W_y)\n",
    "    lst = lst*power\n",
    "    Hf = reduce(np.kron, lst)\n",
    "    w_x, w_y = Hf[W_x], Hf[W_y]\n",
    "    for i, h in enumerate(w_x):\n",
    "        tupleprdt.append((h, w_y[i]))\n",
    "    tupleprdt = WF_Conditions(tupleprdt)\n",
    "    return tupleprdt\n",
    "print(WF_Generate(params, W_x = 1, W_y = 2))\n",
    "    \n",
    "def WF_WIList(params, **kwargs):\n",
    "    W_x, W_y, tupleprdt, ps = kwargs['W_x'], kwargs['W_y'], [], [[]]\n",
    "    for i, w_x in enumerate(W_x):\n",
    "        tupleprdt.append(WF_Generate(params, W_x = w_x, W_y = W_y[i]))\n",
    "    ps = [[]]*len(max(tupleprdt,key=len))\n",
    "    for i, wps in enumerate(tupleprdt):\n",
    "        for j, w in enumerate(wps):\n",
    "            ps.append()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "478cd0f3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1  1]\n",
      " [ 1 -1]]\n"
     ]
    }
   ],
   "source": [
    "lst = [H] # Check how to create arrays of matrices of same type and use it for creating Walsh Functions\n",
    "print(reduce(np.kron, lst))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "dd68a2de",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[], [], []]\n"
     ]
    }
   ],
   "source": [
    "lst = [[]]*3\n",
    "print(lst)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c0c1ca55",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.array(list(itertools.zip_longest(*alist, fillvalue='dummy'))).T"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

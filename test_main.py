#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:34:40 2018

@author: desktop
"""

# Test voltage input in order to remove single scripts put everything in function

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import functions.system_tools as st
from scipy.optimize import minimize
import scipy

from functions.impedance_functions import createImpedanceVoltage
from functions.solver import createAxis

#if __name__ == "__main__":



#%% Create Setup

T = 300 # temperature
epsilon_m = 80 # dielectric constant of water
L = 20 # electrode separation
lengthscalefactor = 1e9 # in nm
c0 = 1 * 1e3 * st._AVOGADRO_CONST / (lengthscalefactor**3) # reference concentration
phi0 = st._BOLTZMANN_CONST * T / st._ELE_CHARGE # thermal energy

# normed t axis, so tmax [s] = tmax x T0
Dt = 0.00025
N =  2**19
t_ = np.zeros(N)
for j in range(0,N):
    t_[j] = j*Dt

# calculation of numerical constants
D0 = 1e-12 * (lengthscalefactor**2)
T0 = L**2 / D0 # reference time
f0 = L*c0/T0 # calculate reference flux density

# calculation of numerical constants
chi1 = st.calcChi1( phi0, T )
chi2 = st.calcChi2(L, c0, phi0, epsilon_m)*lengthscalefactor

# computational axis, defines continous values and Dx
I = 2**8 +2
x_, x = createAxis( I )

# permittivity and diffusion constants
epsilon = np.ones(x_.shape)
DC = np.ones(x_.shape)
DA = np.ones(x_.shape)

print("Min Freq, Frequency Resolution Df [Hz]: ", 1/(t_.size*Dt*T0))
print("Min Freq, Frequency Resolution Df [None]: ", 1/(t_.size*Dt))
print("Maximal Frequency, Nyquist [Hz]: ", 1/(T0*2.2*Dt))
print("Maximal Frequency, Nyquist [None]: ", 1/(2.2*Dt))
print("Number of Points: ", N)

#%%


voltage = createImpedanceVoltage(N, Dt, T0, phi0, U_offset = 0, num = 40)

#%% Call solver function

#%% Test residual function
import numpy as np
import functions.solver_tools as solt
import functions.system_tools as st
from functions.solver import createAxis
from new_solver_test import residual_model0_new

#%%

T = 300 # temperature
epsilon_m = 80 # dielectric constant of water
L = 20 # electrode separation
lengthscalefactor = 1e9 # in nm
c0 = 1 * 1e3 * st._AVOGADRO_CONST / (lengthscalefactor**3) # reference concentration
phi0 = st._BOLTZMANN_CONST * T / st._ELE_CHARGE # thermal energy

# normed t axis, so tmax [s] = tmax x T0
Dt = 0.00025
N =  2**19
t_ = np.zeros(N)
for j in range(0,N):
    t_[j] = j*Dt

# calculation of numerical constants
D0 = 1e-12 * (lengthscalefactor**2)
T0 = L**2 / D0 # reference time
f0 = L*c0/T0 # calculate reference flux density

# calculation of numerical constants
chi1 = st.calcChi1( phi0, T )
chi2 = st.calcChi2(L, c0, phi0, epsilon_m)*lengthscalefactor

# computational axis, defines continous values and Dx
I = 2**8 +2
x_, x = createAxis( I )

# permittivity and diffusion constants
epsilon = np.ones(x_.shape)
DC = np.ones(x_.shape)
DA = np.ones(x_.shape)

solN = np.zeros([3*I], dtype = np.float64)
sol1 = np.zeros([3*I], dtype = np.float64) + np.random.rand(3*I)

phiC = 1.0
#%%
%%timeit

res = solt.residual_model0( I, x_, solN, sol1, chi1, chi2, DC, DA, Dt, phiC, epsilon )

#%%
%%timeit

res1 = residual_model0_new( I, x_, solN, sol1, chi1, chi2, DC, DA, Dt, phiC, epsilon )






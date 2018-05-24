#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:34:13 2018

@author: desktop
"""

# import libraries
from functions.solver import ImpedanceSolver_m0
from functions.read_setup import setup
from functions.impedance_functions import createImpedanceVoltage
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#if __name__ == "__main__":

# import setup
path_2_setupfile = r"test_setup.txt"

stp = setup( path_2_setupfile )


#%% Create voltage pertubation

N = 2024
M = 1e3
Dt = 1e-3
T0 = M * stp.L ** 2 / stp.D0

# create time axis
time = np.zeros(N, dtype = np.float64)
for j in range(0,N):
    
    time[j] = j * Dt
    
    
# frequency in Hz and amplitude [mV]
freq = 10
U0 = 20


phiC = np.zeros( N, dtype = np.float64)

phiC = (U0 * 1e-3) * np.sin( 2 * np.pi * freq * T0 * time) / stp.phi0
    
fig = plt.figure(dpi = 130)

ax = fig.add_subplot(1,1,1)

ax.plot(time, phiC, color = "blue")

plt.show()
    
#%%

# initial condition
sol_initial = np.zeros([3 * stp.I], dtype = np.float64)
sol_initial[0:2*stp.I] = 1.0 / 2.0

# call solver
current = ImpedanceSolver_m0( stp.I,
                             stp.L,
                             stp.c0,
                             stp.epsilon_m,
                             stp.phi0,
                             stp.T,
                             N,
                             Dt,
                             M,
                             sol_initial,
                             phiC,
                             stp.DC,
                             stp.DA,
                             stp.epsilon,
                             stp.model,
                             stp.kA,
                             stp.foxA,
                             stp.kC,
                             stp.foxC )

    # dont save because its a test file

#%% fitting
    
def fit_fun( time, y0, k0, k1 ):
    global freq, T0
    
    return y0 + k0 * np.sin( 2 * np.pi * freq * time *T0 ) + k1 * np.cos( 2 * np.pi * freq * time *T0 )

p_initial = [ U0+1, np.pi]

popt, pcov = curve_fit( fit_fun, time[1:], current[0,1:], p0 = None  )

# calculate parameter
A = np.sqrt( popt[1] ** 2 + popt[2] ** 2 )
phase = np.arctan( popt[2] / popt[1] )

print("A", A)
print("phase", np.rad2deg(shift))

#%%
# plotting

fig = plt.figure(dpi = 130)

ax = fig.add_subplot(1,1,1)

ax.plot([0, time[-1]], [A,A])
ax.plot([0, time[-1]], [-A,-A])
ax.plot(time[1:], phiC[1:], color = "blue")
ax.plot(time[1:], current[0,1:], color = "red")

# plot fitted curve
ax.plot(time, fit_fun( time, popt[0], popt[1], popt[2] ), ls = "--", color =  "black")

plt.show()

print(popt)

print(pcov)

fig = plt.figure(dpi = 130)

ax = fig.add_subplot(1,1,1)

ax.plot(phiC, current[0,:], color = "blue")
#ax.plot(time, , color = "red")

# plot fitted curve
#ax.plot(time, fit_fun( time, popt[0], popt[1] ), ls = "--", color =  "black")

plt.show()

#%% Calculate complex impedance

Z = np.abs( U0 ) * np.exp(1j * phase) / np.abs(A)


fig = plt.figure(dpi = 130)

ax = fig.add_subplot(1,1,1)

ax.plot(Z.real, Z.imag, color = "blue", marker = ".",)
#ax.plot(time, , color = "red")

# plot fitted curve
#ax.plot(time, fit_fun( time, popt[0], popt[1] ), ls = "--", color =  "black")

plt.show()


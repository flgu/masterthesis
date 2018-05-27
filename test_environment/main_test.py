#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import libraries
from functions.solver import ImpedanceSolver
from functions.read_setup import setup
import functions.system_tools as st

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def fitCurrent( independent, y0, k0, k1 ):
    """
    Independent input parameters, list:
        
        independent[0] = time axis
        independent[1] = frequency
        independent[2] = reference time T0
        
    """
    
    time = independent[0]
    freq = independent[1]
    T0 = independent[2]
    
    return y0 + k0 * np.sin( 2 * np.pi * freq * time *T0 ) + k1 * np.cos( 2 * np.pi * freq * time *T0 )

#%%

# import setup
path_2_setupfile = r"test_setup.txt"

stp = setup( path_2_setupfile )


# choose file name
testname = r"model0_series0.txt"


    
    
    

#%% Create voltage pertubation beginning and determine frequency range

stp.I = 400

N = 20
#M = 1
Dt = 1e-3
T0 = stp.L ** 2 / stp.D0
f0 = stp.c0 * stp.D0 / (stp.L)

# create time axis
t = np.zeros(N, dtype = np.float64)
for j in range(0,N):
    
    t[j] = j * Dt
    
    
# frequency in Hz and amplitude [mV]
freq = 1e8
U0 = 10


phiC = np.zeros( N, dtype = np.float64)

phiC =  np.ones(N, dtype = np.float64) * 10e-3 / stp.phi0#(U0 * 1e-3) * np.sin( 2 * np.pi * freq * T0 * t) / stp.phi0
    
fig = plt.figure(dpi = 130)

ax = fig.add_subplot(1,1,1)

ax.plot(t, phiC, color = "blue")

plt.show()
    
#%% loop over frequency range

# initial condition
sol_initial = np.zeros([3 * stp.I], dtype = np.float64)
sol_initial[0:2*stp.I] = 1.0 / 2.0

# delete existing previous current output
try:

    del outlist
    
except:
    
    pass

# returnmode
returnmode = "full_solution"

# call solver
outlist = ImpedanceSolver( stp.I,
                             stp.L,
                             stp.c0,
                             stp.epsilon_m,
                             stp.phi0,
                             stp.T,
                             N,-
                             Dt,
                             #M,
                             sol_initial,
                             phiC,
                             stp.DC,
                             stp.DA,
                             stp.epsilon,
                             stp.model,
                             stp.kA,
                             stp.foxA,
                             stp.kC,
                             stp.foxC,
                             returnmode = returnmode)

#%%

index = 1

sol = outlist[0]

fig = plt.figure(dpi = 130)

ax = fig.add_subplot(1,1,1)

ax.plot(sol[0:stp.I,index])

plt.show()



#%%
# fitting

if not p_initial.__len__() == 0:
    
    print("using initial fit conditions from previous simulation")
    
    popt, pcov = curve_fit( fitCurrent, [t[1:],freq,T0] , current[0,1:], p0 = p_initial  )
    
else:
    popt, pcov = curve_fit( fitCurrent, [t[1:],freq,T0], current[0,1:], p0 = None  )
    
    p_initial = popt


# calculate parameters for current
A = np.sqrt( popt[1] ** 2 + popt[2] ** 2 ) * f0 * st._ELE_CHARGE * 1e17
phase = np.arctan( popt[2] / popt[1] )

# calculate impedance
Z = np.abs( U0 * stp.phi0 ) * np.exp(1j * phase) / np.abs(A)

print("A", A)
print("phase", np.rad2deg(phase))
print("Impedance", Z)

# saving impedance results

savearray = np.array([[ freq, Z  ]])


#%%
# plotting

fig = plt.figure(dpi = 130)

ax = fig.add_subplot(1,1,1)


ax.plot(t[1:], phiC[1:], color = "blue")
ax.plot(t[1:], current[0,1:], color = "red")

# plot fitted curve
ax.plot(t, fitCurrent( [t,freq,T0], popt[0], popt[1], popt[2] ), ls = "--", color =  "black")

#ax.plot([0, t[-1]], [A,A])
#ax.plot([0, t[-1]], [-A,-A])

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



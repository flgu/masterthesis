#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:00:01 2018

@author: florian

Testing upwinding implementation in module res.
Compare solutions (res.pyd = old implementation)

"""
#%%
import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import time
import matplotlib.pyplot as plt
from solver import PDEsolver
import system_tools as st



#%% Create Setup

T = 300
epsilon_m = 80 # dielectric constant of water
L =  10 # electrode separation
lengthscalefactor = 1e9 # in nm
c0 = 1 * 1e3 * st._AVOGADRO_CONST / (lengthscalefactor**3) # reference concentration
zC = 1
zA = -1
phi0 = st._BOLTZMANN_CONST * T / st._ELE_CHARGE # thermal energy
D0 = 1e-13 * (lengthscalefactor**2)

# calculation of numerical constants
T0 = L**2 / D0 # reference time
f0 = L*c0/T0 # calculate reference flux density
chi1 = st.calcChi1( phi0, T )
chi2 = st.calcChi2(L, c0, phi0, epsilon_m)*lengthscalefactor

# time axis calculation
# normed t axis, so tmax [s] = tmax x T0
Dt = 0.001
N =  2**10

# computational axis, defines continous values and Dx
I = 2**9 +2
x_ = np.zeros(I+1)
Dx = 1/(I)
for i in range(0,I+1):
    x_[i] = i*Dx

# coordinates of cell centers
x = np.zeros(I)
for i in range(1,I+1):
    x[i-1] = x_[i-1]+(x_[i]-x_[i-1])/2

DC = np.ones(x_.shape)* 1e-11 * (lengthscalefactor**2) / D0
DA = np.ones(x_.shape)* 1e-11 * (lengthscalefactor**2) / D0
epsilon = np.ones(x_.shape)

# reaction rates --> just use blocking electrode
bctype = "linear"
jC0 = 0.0
kC0 = 0.0
jCI = 0.0
kCI = 0.0

#%% sinuidal voltage pertubation (long t)
# calculate t_ axis
t_ = np.zeros(N)
for j in range(0,N):
    t_[j] = j*Dt

g0_ = np.zeros(t_.size); gI_ = np.zeros(t_.size)
freq = T0 * 1e3
ampl = 0.01 / phi0

gI_[:] = ampl * np.sin(2*np.pi*freq*t_)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(t_, gI_)
plt.show()


#%% Test Case for function PDE solver

sol_initial = np.zeros(3*I, dtype = np.float64)
sol_initial[0:2*I] = 1/2

sol, tout_, gIout_ = PDEsolver( I, x_, t_, sol_initial, gI_, DC, DA, epsilon, chi1, chi2, Dt)



#%% Plotting Results
k = -1
plt.close("all")


fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)

# grids
ax.grid(b = True, which = "major", axis = "both")

# plot concentration cells
ax.plot(x, sol[0:I,k], color = "blue")
ax.plot(x, sol[I:2*I,k], color = "red")

ax1 = ax.twinx()

ax1.plot(x, sol[2*I:3*I,k], color = "black")

plt.show()

#%%

int_c = np.sum(sol[0:I,:]*Dx, axis = 0)

fig2 = plt.figure()
ax = fig2.add_subplot(1,1,1)

ax.plot(t_, int_c, color = "blue")

fig2.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:34:40 2018

@author: desktop
"""

# Test voltage input in order to remove single scripts put everything in function

import numpy as np
import matplotlib.pyplot as plt

import functions.system_tools as st

#if __name__ == "__main__":
#%%
#%% Create Setup
T = 300
epsilon_m = 80 # dielectric constant of water
L =  10 # electrode separation
lengthscalefactor = 1e9 # in nm
c0 = 1 * 1e3 * st._AVOGADRO_CONST / (lengthscalefactor**3) # reference concentration
phi0 = st._BOLTZMANN_CONST * T / st._ELE_CHARGE # thermal energy

# time axis calculation
# normed t axis, so tmax [s] = tmax x T0
Dt = 0.001
N =  2**19

D0 = 1e-12 * (lengthscalefactor**2)

# calculation of numerical constants
T0 = L**2 / D0 # reference time
f0 = L*c0/T0 # calculate reference flux density

t_ = np.zeros(N)
for j in range(0,N):
    t_[j] = j*Dt

print("Min Freq, Frequency Resolution Df [Hz]: ", 1/(t_.size*Dt*T0))
print("Min Freq, Frequency Resolution Df [None]: ", 1/(t_.size*Dt))
print("Maximal Frequency, Nyquist [Hz]: ", 1/(T0*2.2*Dt))
print("Maximal Frequency, Nyquist [None]: ", 1/(2.2*Dt))
print("Number of Points: ", N)

#%%
f_Ny = np.floor(1.0 / (2.2 * Dt)) # Maximal frerquency with secturity 2.2, floor that
f_s = 1.0 / (t_.size * Dt)

factors = np.array([2, 4, 6, 8])

print(np.floor(f_Ny / f_s))


fac = np.concatenate( (factors,np.geomspace(10,np.floor(f_Ny / f_s),num = 40)) )


voltage = np.zeros(N, dtype = np.float64)

U0 = 0

for i in range(0, fac.size):

	freq = f_s * int(fac[i])

	voltage += np.sin( 2 * np.pi * freq * t_ )

voltage += U0

Fvoltage = np.fft.fft(voltage, n = N)[1:int(N/2)]
freq_ax = np.fft.fftfreq(N, d = Dt)[1:int(N/2)]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(freq_ax, Fvoltage.imag, color = "red")

ax.set_xscale("log")

plt.show()

amplitude = np.abs( voltage.max() - voltage.min() )
print(amplitude * phi0 * 1e3)











#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:00:01 2018

@author: florian

Testing upwinding implementation in module res.
Compare solutions (res.pyd = old implementation)

"""


import ..functions.system_tools as st
#%%
import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import time
import matplotlib.pyplot as plt
import system_tools as st

#%% Create Setup
T = 300
epsilon_m = 80 # dielectric constant of water
L =  10 # electrode separation
lengthscalefactor = 1e9 # in nm
c0 = 1 * 1e3 * st._AVOGADRO_CONST / (lengthscalefactor**3) # reference concentration
phi0 = st._BOLTZMANN_CONST * T / st._ELE_CHARGE # thermal energy

# computational axis, defines continous values and Dx
I = 2**9 +2

# reaction rates --> just use blocking electrode
jC0 = 0.0
kC0 = 0.0
jCI = 0.0
kCI = 0.0


#%% Analyse sampling

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
print("Maximal Frequency, Nyquist [Hz]: ", 1/(T0*2.5*Dt))
print("Maximal Frequency, Nyquist [None]: ", 1/(2.5*Dt))
print("Number of Points: ", N)


#%% sinuidal voltage pertubation (long t)
freq_list = [3,  6,  8, 9, 30, 40,  60,  80, 90, 300, 500, 600, 700, 800, 900, 1500, 3000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 13000, 20000,
        30000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000, 209700,210000, 220000,
       230000, 240000, 250000, 255000 ]
ampl = 0.00002
U0 = 0.0

gI_ = st.sineSeries( t_, Dt, T0, U0, ampl,  freq_list, phi0)

Fvoltage = np.fft.fft(gI_)[1:int(N/2)]
frq_ax = np.fft.fftfreq(N,d = Dt)[1:int(N/2)]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xscale("log")
ax.plot(frq_ax, Fvoltage.imag)
plt.show()

#%% Save voltage input and setup file
np.save("voltage_input.npy", gI_)

setup_dict = {
    "ref_concentration": c0,
    "ref_potential": phi0,
    "ref_diffusion": D0,
    "lengthscalefactor": lengthscalefactor,
    "Dt": Dt,
    "N": N,
    "I":I,
    "jC0":jC0,
    "kCI":kCI,
    "kC0":kC0,
    "jCI":jCI
    }
np.save("setup_input.npy",setup_dict)

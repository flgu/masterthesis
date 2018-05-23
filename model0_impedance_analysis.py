
#%% Diffusion / Impedance Test Script
machine = "work"

import os 

# import
import numpy as np
import time

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rc('text', usetex=False)
plt.rc('font', family='serif')

from functions.read_setup import setup
import functions.system_tools as st
from functions.impedance_functions import createImpedanceVoltage

#%% Load files


# import last valid setup.txt; has to be updated with values in simulation plan
path_2_setup = r"setup.txt"
stp = setup( path_2_setup )


#%% Load different current solutions from impedance spectroscopy

# Dt and N for all equal --> Difference in L
L = 90

# calculate T0 and f0 for this simulation
T0 = L ** 2 / stp.D0
f0 = L * stp.c0 / T0

path_2_sol_1 = r"solutions/impedance_test_3.npy"

current_1 = np.load( path_2_sol_1 ) * f0 * st._ELE_CHARGE * 1e17


voltage = createImpedanceVoltage( stp.N, stp.Dt, T0, stp.phi0, U_offset = 0, num = 40)

Fvoltage = np.fft.fft(voltage)[0:int(voltage.shape[0]/2)]
freq_ax = np.fft.fftfreq(voltage.shape[0],d = 0.001 )[0:int(voltage.shape[0]/2)]

del voltage


#%% FFT of input voltage
FcurrentA = np.fft.fft(current_1[0,:])[0:int( stp.N/2)]
FcurrentC = np.fft.fft(current_1[1,:])[0:int( stp.N/2)] 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(freq_ax[1:] / T0, FcurrentA.imag[1:], color = "red")
ax.plot(freq_ax[1:] / T0, FcurrentA.real[1:], color = "blue")


# ticks formatting
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%1.1e"))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
#ax.yaxis.set_minor_locator(ticker.MultipleLocator())

# grid
ax.grid(b = True, which = "major", axis = "both")
ax.grid(b = True, which = "minor", axis = "both")

# labels
ax.set_xlabel(r"f [Hz]")
ax.set_ylabel(r"Imag $\left[ \Omega cm^2 \right]$")

plt.show()

# Impedance calculation
Z_A = Fvoltage / FcurrentA


#%% Plotting

# Nyquist Plot current
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(FcurrentA.real[1:], FcurrentA.imag[1:])

plt.show()


#%% Impedance Nyquist
saveflag = 0

fig = plt.figure()
ax = fig.add_subplot(1,1,1)



ax.plot(np.abs(Z_A.real[:]), np.abs(Z_A.imag[:]), ls = "None", marker = ".", markersize = 8)

#ax.set_xlim([0 , 1e-5])

# ticks formatting
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%1.1e"))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))

# grid
ax.grid(b = True, which = "major", axis = "both")
ax.grid(b = True, which = "minor", axis = "both")

# labels
ax.set_xlabel(r"Real $\left[ \Omega cm^2 \right]$")
ax.set_ylabel(r"Imag $\left[ \Omega cm^2 \right]$")

if saveflag == 1:
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(figs_path, fig_name + "." +fmt ), format = fmt, dpi = 300)

plt.show()
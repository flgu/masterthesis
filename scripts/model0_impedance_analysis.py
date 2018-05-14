
#%% Diffusion / Impedance Test Script
machine = "work"

import os 

if machine == "home":
    codepath = r"D:\masterthesis\implementations\model_1\code"
    solutions_path = r"D:\masterthesis\implementations\model_1\solutions"
    input_files_path = r"D:\masterthesis\implementations\model_1\input_files"
elif machine == "work":
    codepath = r"M:\QMR\Abteilungsprojekte\FG\008_masterthesis\implementations\model_1\code"
    solutions_path = r"M:\QMR\Abteilungsprojekte\FG\008_masterthesis\implementations\model_1\solutions"
    figs_path = r"M:\QMR\Abteilungsprojekte\FG\008_masterthesis\implementations\model_1\solutions\figs"
    input_files_path = r"M:\QMR\Abteilungsprojekte\FG\008_masterthesis\implementations\model_1\input_files"

os.chdir(codepath)

print("Current working directory :", os.getcwd())

# import
import numpy as np
import time

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rc('text', usetex=False)
plt.rc('font', family='serif')

from solver import ImpedanceSolver, createAxis
import system_tools as st

#%% Load files

T0 = 0.0001
D0 = 1000000
Dt = 0.001
L = 10
T = 300
lengthscalefactor = 1e9 # in nm
c0 = 1 * 1e3 * st._AVOGADRO_CONST / (lengthscalefactor**3)
f0 = L*c0/T0
phi0 = st._BOLTZMANN_CONST * T / st._ELE_CHARGE # thermal energy

voltage = np.load(os.path.join(r"M:\QMR\Abteilungsprojekte\FG\008_masterthesis\implementations\model_1\input_files\voltage_input.npy")) * phi0

current = (np.load(os.path.join(r"M:\QMR\Abteilungsprojekte\FG\008_masterthesis\implementations\model_1\solutions\current_impedance_s0l\current_impedance_s0l.npy")) * f0 * st._ELE_CHARGE * 1e14)

# FFT

Fvoltage = np.fft.fft(voltage)[0:int(voltage.shape[0]/2)]
FcurrentA = np.fft.fft(current[0,:])[0:int(voltage.shape[0]/2)]
FcurrentC = np.fft.fft(current[1,:])[0:int(voltage.shape[0]/2)]

freq_ax = np.fft.fftfreq(voltage.shape[0],d = 0.001 )[0:int(voltage.shape[0]/2)]

# Impedance calculation
Z_A = Fvoltage / FcurrentA

#%% Plotting

# Nyquist Plot current
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(FcurrentA.real[1:], FcurrentA.imag[1:])

plt.show()

#%% FFT vs freq
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(freq_ax[1:], FcurrentA.imag[1:], color = "red")
ax.plot(freq_ax[1:], FcurrentA.real[1:], color = "blue")

plt.show()

#%% Impedance Nyquist
figs_path = r"M:\QMR\Abteilungsprojekte\FG\008_masterthesis\implementations\model_1\solutions\current_impedance_s0l"
fig_name = r"impedance_nyquist"
saveflag = 1

fig = plt.figure()
ax = fig.add_subplot(1,1,1)



ax.plot(np.abs(Z_A.real[:]), np.abs(Z_A.imag[:]), ls = "None", marker = ".", markersize = 8)



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
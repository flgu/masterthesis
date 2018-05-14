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


#%% Create Setup

#if __name__ == "__main__":
    
T = 300 # temperature
epsilon_m = 80 # dielectric constant of water
L = 10 # electrode separation
lengthscalefactor = 1e9 # in nm
c0 = 1 * 1e3 * st._AVOGADRO_CONST / (lengthscalefactor**3) # reference concentration
phi0 = st._BOLTZMANN_CONST * T / st._ELE_CHARGE # thermal energy

# normed t axis, so tmax [s] = tmax x T0
Dt = 0.001
N =  2**19
t_ = np.zeros(N)
for j in range(0,N):
    t_[j] = j*Dt

# calculation of numerical constants
D0 = 1e-12 * (lengthscalefactor**2)
T0 = L**2 / D0 # reference time
f0 = L*c0/T0 # calculate reference flux density

# reaction rates --> just use blocking electrode
kA = 0.1
kC = 0.10
foxA = 0.05
foxC = 0.05

# calculation of numerical constants
chi1 = st.calcChi1( phi0, T )
chi2 = st.calcChi2(L, c0, phi0, epsilon_m)*lengthscalefactor
    
# computational axis, defines continous values and Dx
I = 2**9 +2
x_, x = createAxis( I )

# permittivity and diffusion constants
epsilon = np.ones(x_.shape)
DC = np.ones(x_.shape) * D0 / D0
DA = np.ones(x_.shape) * D0 / D0

#%% Simulation loop

# initiate intial condition

sol_initial = np.load(input_files_path + r"\impedance_input_1.npy")
phiC =  np.load(input_files_path + r"\voltage_input.npy")


#%% read in ac voltage pertubation

# calculate t_ axis, maybe cutted from before

    
# call pde solver
current = ImpedanceSolver( I, x_, t_, sol_initial, phiC, DC, DA, epsilon, chi1, chi2, Dt, kA, kC, foxA, foxC)

np.save( os.path.join(r"M:\QMR\Abteilungsprojekte\FG\008_masterthesis\implementations\model_1\solutions\current_impedance_s02","current_impedance_s02.npy"),current)


#%% Diffusion / Impedance Test Script
machine = "home"

import os
import imp

if machine == "home":
    functionspath = r"D:\masterthesis\implementations\functions"
    codepath = r"D:\masterthesis\implementations\model_0\code"
    solutions_path = r"D:\masterthesis\implementations\model_0\solutions"
    input_files_path = r"D:\masterthesis\implementations\model_0\input_files"
    
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

(file, pathname, description) = imp.find_module("solver",
 [os.path.join(functionspath)])

solver = imp.load_module("solver",file, pathname, description)

(file, pathname, description) = imp.find_module("system_tools",
 [os.path.join(functionspath)])

st = imp.load_module("system_tools",file, pathname, description)



#%% Create Setup

#if __name__ == "__main__":
    
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
x_, x = solver.createAxis( I )

# permittivity and diffusion constants
epsilon = np.ones(x_.shape)
DC = np.ones(x_.shape)
DA = np.ones(x_.shape)

#%% Simulation loop

# initiate intial condition

sol_initial = np.zeros(3*I, dtype = np.float64)
sol_initial[0:2*I] = 1.0 / 2.0

phiC =  np.load(input_files_path + r"\voltage_input.npy")


#%% read in ac voltage pertubation

# calculate t_ axis, maybe cutted from before

    
# call pde solver
current = solver.ImpedanceSolverModel0( I, x_, t_, sol_initial, phiC, DC, DA, epsilon, chi1, chi2, Dt )

np.save( os.path.join(r"D:\masterthesis\implementations\model_0\solutions\impedance_L_1","current_impedance_1.npy"),current)


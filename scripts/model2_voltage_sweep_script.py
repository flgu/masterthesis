#%% Diffusion / Impedance Test Script
import os 

winpath = r"D:\masterthesis\implementations\model_2\reaction_boundary_conditions"

os.chdir(winpath)

print("Current working directory :", os.getcwd())

# import
import numpy as np
import time
import matplotlib.pyplot as plt
from solver import SweepSolver
import system_tools as st

#%% Create Setup
if __name__ == "__main__":

    testname = "desktop_E_sweep_" 

    T = 300 # temperature
    epsilon_m = 80 # dielectric constant of water
    L = 10 # electrode separation
    lengthscalefactor = 1e9 # in nm
    c0 = 1 * 1e3 * st._AVOGADRO_CONST / (lengthscalefactor**3) # reference concentration
    phi0 = st._BOLTZMANN_CONST * T / st._ELE_CHARGE # thermal energy

    # normed t axis, so tmax [s] = tmax x T0
    Dt = 0.0005
    N =  2**13
    t_ = np.zeros(N)
    for j in range(0,N):
        t_[j] = j*Dt

    # calculation of numerical constants
    D0 = 1e-12 * (lengthscalefactor**2)
    T0 = L**2 / D0 # reference time
    f0 = L*c0/T0 # calculate reference flux density


    # reaction rates --> just use blocking electrode
    kA = 1.0
    kC = 1.0
    foxA = 0.5
    foxC = 0.5

    # Equilibrium energies in [mV]
    E_A_vec = np.array([10, 20, 30], dtype = np.float64) * 1e-3
    E_C_vec = np.array([10, 20, 30], dtype = np.float64) * 1e-3

    # calculation of numerical constants
    chi1 = st.calcChi1( phi0, T )
    chi2 = st.calcChi2(L, c0, phi0, epsilon_m)*lengthscalefactor
    
    # computational axis, defines continous values and Dx
    I = 2**9 +2
    x_ = np.zeros(I+1)
    Dx = 1.0/I
    for i in range(0,I+1):
        x_[i] = i*Dx
    
    # coordinates of cell centers
    x = np.zeros(I)
    for i in range(1,I+1):
        x[i-1] = x_[i-1]+(x_[i]-x_[i-1])/2

    # permittivity and diffusion constants vector (for looping)
    epsilon = np.ones(x_.shape)
    DC_vec = np.array([1, 5, 15, 20, 10, 15, 20], dtype = np.float64)
    DA_vec = np.array([1,  1,  1,  1, 10, 10, 10], dtype = np.float64) 

    # voltage vector [mV] 
    voltage = np.arange(-100,100,10, dtype = np.float64)

    #%% Simulation loop

    # initiate intial condition
    sol_initial = np.zeros(3*I, dtype = np.float64)
    sol_initial[0:2*I] = 1/2

    # outer loop
    for iterK in range(0,DA_vec.size):

        # diffusion constants
        DC = np.ones(x_.shape) * DC_vec[0]
        DA = np.ones(x_.shape) * DA_vec[0]

        # sweep output matrix with current densities
        out_matrix = np.zeros([voltage.size, 3], dtype = np.float64)
        out_matrix[:,0] = voltage[:]

        # boundary equilibrium energies
        E_A = E_A_vec[iterK] / phi0
        E_C = E_C_vec[iterK] / phi0

        # inner loop through voltages
        for iterI in range(0,voltage.size):
    
            print("testnummer:",iterI)
   
            # boundary condition
            phiC = np.ones(t_.shape, dtype = np.float64) * 1e-3 * voltage[iterI] / phi0
    
            # call pde solver
            sol, current = SweepSolver( I, x_, t_, sol_initial, phiC, DC, DA, epsilon, chi1, chi2, Dt, kA, kC, foxA, foxC, E_A, E_C,)

            # extract steady state current
            out_matrix[iterI,1] = current[0,-1]
            out_matrix[iterI,2] = current[1,-1]

            del sol, current

        np.save(testname + str(iterK), out_matrix)

        del out_matrix
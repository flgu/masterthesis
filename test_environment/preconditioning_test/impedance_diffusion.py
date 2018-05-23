# Diffusion / Impedance Test Script

#%% import
import numpy as np
import time
import matplotlib.pyplot as plt
from solver import PDEsolver
import system_tools as st

#%% choose directories for saving
directory = r"../model_0_noflux/imp_spec/"

# read in setup file
read_in = np.load(directory + r"imp_D2_input.npy").item()

testname  = "imp_D2"
T = read_in["temperature"] # temperature
epsilon_m = read_in["epsilon_m"] # dielectric constant of water
L = read_in["length"] # electrode separation
c0 = read_in["ref_concentration"] # reference concentration
zC = read_in["valency_C"]
zA = read_in["valency_A"]
phi0 = read_in["ref_potential"]
D0 = read_in["ref_diffusion"]
lengthscalefactor = read_in["lengthscalefactor"]
T0 = L**2 / D0 # reference time
f0 = L*c0/T0 # calculate reference flux density

# calculation of numerical constants
chi1 = res.calcChi1( phi0, T )
chi2 = res.calcChi2(L, c0, phi0, epsilon_m)*lengthscalefactor

# time axis calculation
# normed t axis, so tmax [s] = tmax x T0
Dt = read_in["Dt"]
N =  read_in["N"]
t_ = np.zeros(N)
for j in range(0,N):
    t_[j] = j*Dt
    
# computational axis, defines continous values and Dx
I =  read_in["I"]
x_ = np.zeros(I+1)
Dx = 1/(I)
for i in range(0,I+1):
    x_[i] = i*Dx
    
# coordinates of cell centers
x = np.zeros(I)
for i in range(1,I+1):
    x[i-1] = x_[i-1]+(x_[i]-x_[i-1])/2
    
jC0 =  read_in["jC0"]
kCI =  read_in["kCI"]
kC0 =  read_in["kC0"]
jCI = read_in["jCI"]

DC = np.ones(x_.shape)*read_in["cation_diff"] / D0
DA = np.ones(x_.shape)*read_in["anion_diff"] / D0
epsilon = np.ones(x_.shape)

#%% loop over diffusion values

# initiate intial condition
sol_initial = np.zeros(3*I, dtype = np.float64)
sol_initial[0:2*I] = 1/2


for iterI in range(0,1):
    
    print("testnummer:",iterI)
   
    # calculate t_ axis, maybe cutted from before
    if t_.size < N:
        t_ = np.zeros(N)
        for j in range(0,N):
            t_[j] = j*Dt
    
    gI_ = np.zeros(t_.size)

    
    
    # initialize solution vector
    sol = np.zeros([3*x.size,t_.size])
    
    if loadflag == 0:
    
        # initial conditions
        sol[0:I,0] = 1/2
        sol[I:2*I,0] = 1/2
        
        # voltage as step_function
        gI_[1:] = volt_[iterI]
        
    elif loadflag == 1:
        
        gI_[:] = np.load(testname + "voltage_input.npy", mmap_mode=None,
        allow_pickle=True, fix_imports=False, encoding='ASCII')
        
        sol1 = np.load("impedance_test_1_imp__sol.npy", mmap_mode=None,
        allow_pickle=True, fix_imports=False, encoding='ASCII')
        
        sol[:,0] = sol1[:,-1]
        
        del sol1
    elif loadflag == "impedance":
        
        # initial conditions
        sol[0:I,0] = 1/2
        sol[I:2*I,0] = 1/2
        
        gI_[:] = np.load(directory + testname + "_voltage_input.npy", mmap_mode=None,
        allow_pickle=True, fix_imports=False, encoding='ASCII')
        
    if loadflag == "volt_sweep":
        
        # initial conditions
        sol[0:I,0] = 1/2
        sol[I:2*I,0] = 1/2
        gI_[:] = volt_[iterI]
        
          
    t1 = time.clock()

    # call pde solver
    
    # save results in specific directory

    # delete solution
    t2 = time.clock()
    print("Simulation Runtime: ", t2-t1)
            
        
    np.save(directory + testname + "_sol_"+ str(iterI) +".npy", sol, allow_pickle=True, fix_imports=False)    
    
    # calc and save current and voltage from DL
#    current, voltage = calc_DL_Volt_Cur(t_, sol, g0_, gI_, Dt, chi2, bctype, kC0, jC0, kCI, jCI)
#    
#    np.save(testname + "_volt_"+ str(iterI) +".npy", voltage, allow_pickle=True, fix_imports=False)    
#    np.save(testname + "_cur_"+ str(iterI) +".npy", current, allow_pickle=True, fix_imports=False)  
#%% Diffusion / Impedance Test Script
import os 

path = r"D:\masterthesis\implementations\model_2\reaction_boundary_conditions"

os.chdir(path)

print("Current working directory :", os.getcwd())

# import
import numpy as np
import time
import matplotlib.pyplot as plt
from solver import PDEsolver
import system_tools as st

#%% Create Setup

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
E_A = 0.001
E_C = 0.001

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


# permittivity and diffusion constants
epsilon = np.ones(x_.shape)
DC = np.ones(x_.shape) * D0 / D0
DA = np.ones(x_.shape) * D0 / D0

# voltage vector
voltage = np.array([0.0], dtype = np.float64)

#%% Simulation loop

# initiate intial condition
sol_initial = np.zeros(3*I, dtype = np.float64)
sol_initial[0:2*I] = 1/2

# read in ac voltage pertubation


for iterI in range(0,1):
    
    print("testnummer:",iterI)
   
    # calculate t_ axis, maybe cutted from before
    phiC = np.ones(t_.shape, dtype = np.float64) * voltage[iterI] / phi0
    
    # call pde solver
    sol = PDEsolver( I, x_, t_, sol_initial, phiC, DC, DA, epsilon, chi1, chi2, Dt, kA, kC, foxA, foxC, E_A, E_C,)

#%% Examine Steady State conditions
print(sol[2*I,-1] - E_A)

print(np.log(foxA/(kA * sol[0,-1])))


    # save results in specific directory
    #np.save("sol_"+ str(iterI) +".npy", sol, allow_pickle=True, fix_imports=False)    

#%%
indizes = np.array([-1])

fig = plt.figure(dpi=120)
ax = fig.add_subplot(1,1,1)
ax1 = ax.twinx()

for k in range(0,indizes.size):
    ax.plot(x, sol[0:I,indizes[k]], color = "blue")
    ax.plot(x, sol[I:2*I,indizes[k]], color = "red")

    ax1.plot(x, sol[2*I:3*I,indizes[k]], color = "black")

plt.show()

#%% Überprüfe Randbedingungen
k = -1
alpha_A = 0.5
alpha_C = 0.5

print( foxA * np.exp(- alpha_A*(sol[2*I,k] - E_A) ))
print( - kA * sol[0,k] * np.exp((1 - alpha_A)*(sol[2*I,k] - E_A) ) )

print( kC * sol[I-1,k] * np.exp( - alpha_C*(( phiC[k] - sol[3*I-1,k]) - E_C)   ))
print( - foxC * np.exp( (1 - alpha_C)*( ( phiC[k] - sol[3*I-1,k]) - E_C) )  )



#%% 
fC0 = ( foxA * np.exp(- alpha_A*(solN[2*I] - E_A) ) 
                      - kA * solN[0] * np.exp((1 - alpha_A)*(solN[2*I] - E_A)   ) )
    fCI = (- foxC * np.exp( (1 - alpha_C)*( ( phiC - solN[3*I-1]) - E_C) ) 
           + kC * solN[I-1] * np.exp( - alpha_C*(( phiC - solN[3*I-1]) - E_C)   )  )

#%% 
print(-np.log(sol[I-1,k]/sol[0,k]))
print(sol[3*I-1,k] - sol[2*I,k])
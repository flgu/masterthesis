import numpy as np

import system_tools as st

import c_fun

# old solver tools for comparison
from solver_tools import residual_model0
#%%
from new_solver_test import residual_model0_new

def createAxis(I):

    x_ = np.zeros(I+1)
    Dx = 1.0/I
    for i in range(0,I+1):
        x_[i] = i*Dx

    # coordinates of cell centers
    x = np.zeros(I)
    for i in range(1,I+1):
        x[i-1] = x_[i-1]+(x_[i]-x_[i-1])/2

    return x_, x


# define 1 dim numpy array
arr = np.array([1,2,3,4,5,6], dtype = np.double)
out = c_fun.main( arr, arr )
print(type(out))
print(out)
#%%
%%timeit
out = c_fun.main( arr, arr )

#%%

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
x_, x = createAxis( I )

# permittivity and diffusion constants
epsilon = np.ones(x_.shape)
DC = np.ones(x_.shape)
DA = np.ones(x_.shape)

sol1 = np.ones(3*I, dtype = np.float64) + np.random.rand(3*I)
sol2 = np.ones(3*I, dtype = np.float64) + np.random.rand(3*I)

#%%
%%timeit
residual_model0( I, x_, sol1, sol2, chi1, chi2, DC, DA, Dt, 0.01, epsilon )

#%%
%%timeit
residual_model0_new( I, x_, sol1, sol2, chi1, chi2, DC, DA, Dt, 0.01, epsilon )

#%%
sol_old = residual_model0( I, x_, sol1, sol2, chi1, chi2, DC, DA, Dt, 0.01, epsilon )
sol_new = residual_model0_new( I, x_, sol1, sol2, chi1, chi2, DC, DA, Dt, 0.01, epsilon )


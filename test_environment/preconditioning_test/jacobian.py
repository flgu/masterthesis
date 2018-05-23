"""
For better Newton-Krylov convergence we need to precondition the system.
For that purpose we choose an approximated Jacobian as a right preconditioner.
"""
#%%
import numpy as np
import time
from scipy.optimize import newton_krylov
import res
import matplotlib.pyplot as plt
from residual import residual
import timeit
from jacobian import calcJac
from jacobian_new import calcJac as calcJac_new

#%%
def calcJac(I, sol, x_, DC, DA, chi1, chi2, Dt ):

    M = np.zeros([3*I,3*I], dtype = np.float64)

    # calculate derivatives of boundary condition
    dfAdc = 0.0
    dfAdp = 0.0
    dfCdc = 0.0
    dfCdp = 0.0

    # loop over potential rows and fill jacobian for potential
    for i in range(0,I):

        # fill cation diagonal
        M[2*I+i,i] = chi2

        # fill anion diagonal
        M[2*I+i,I+i] = -chi2

        # fill tridiagonal for inner points
        if i > 0 and i < I-1:

            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # fill lower diag
            M[2*I+i,2*I+i-1] = 1.0/(x_[i+1] - x_[i-1]) * 2.0/(x_[i+1] - x_[i])

            # fill diag
            M[2*I+i,2*I+i] = ( -1.0/(x_[i+2] - x_[i]) - 1.0/(x_[i+1] - x_[i-1]) ) * 2.0/(x_[i+1] - x_[i])

            # fill upper diag
            M[2*I+i,2*I+i+1] = 1.0/(x_[i+2] - x_[i]) * 2.0/(x_[i+1] - x_[i])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # fill lower diag
            M[i,i-1] = Dt * ( -2*DC[i]/(x_[i+1] - x_[i-1]) + DC[i]*chi1*2* np.max([sol[2*I+i] - sol[2*I+i-1],0])/(x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i])
        
            # fill diag
            M[i,i] = ( 1 + Dt * ( 2*DC[i+1]/(x_[i+2] - x_[i]) + 2*DC[i]/(x_[i+1] - x_[i-1]) 
                               - DC[i+1]*chi1*2* np.max([sol[2*I+i+1]- sol[2*I+i],0]) / (x_[i+2] - x_[i])
                               + DC[i]*chi1*2* np.min([sol[2*I+i] - sol[2*I+i-1],0]) / (x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i]) )

            # fill upper diag
            M[i,i+1] = Dt * ( -2*DC[i+1]/(x_[i+2] - x_[i]) - DC[i+1]*chi1*2 * np.min([sol[2*I+i+1] - sol[2*I+i],0]) / (x_[i+2] - x_[i]) ) / (x_[i+1] - x_[i])

            #--- dNP- / dc--------------------------------------------------------------------------------
            # fill lower diag
            M[I+i,I+i-1] = Dt * ( -2*DA[i]/(x_[i+1] - x_[i-1]) - DA[i]*chi1*2* np.max([sol[2*I+i] - sol[2*I+i-1],0])/(x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i])
        
            # fill diag
            M[I+i,I+i] = ( 1 + Dt * ( 2*DA[i+1]/(x_[i+2] - x_[i]) + 2*DA[i]/(x_[i+1] - x_[i-1]) 
                               + DA[i+1]*chi1*2* np.max([sol[2*I+i+1]- sol[2*I+i],0]) / (x_[i+2] - x_[i])
                               - DA[i]*chi1*2* np.min([sol[2*I+i] - sol[2*I+i-1],0]) / (x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i]) )

            # fill upper diag
            M[I+i,I+i+1] = Dt * ( -2*DA[i+1]/(x_[i+2] - x_[i]) + DA[i+1]*chi1*2 * np.min([sol[2*I+i+1] - sol[2*I+i],0]) / (x_[i+2] - x_[i]) ) / (x_[i+1] - x_[i])
            

           

            #--- dNP+- / dp--------------------------------------------------------------------------------
            
            # fill lower diag
            if not (sol[2*I+i] - sol[2*I+i-1]) == 0:
                
                 #--- dNP+ / dp--------------------------------------------------------------------------------
                M[i,2*I+i-1] =( Dt*(-DC[i]*chi1*( sol[i-1]*np.max([sol[2*I+i] - sol[2*I+i-1],0]) 
                            + sol[i]*np.min([sol[2*I+i] - sol[2*I+i-1],0]))*2/((x_[i+1] - x_[i-1]) * (sol[2*I+i] - sol[2*I+i-1])) ) / (x_[i+1] - x_[i]) )
            
                #--- dNP- / dp--------------------------------------------------------------------------------
                M[I+i,2*I+i-1] = ( Dt*(DA[i]*chi1*( sol[I+i-1]*np.max([sol[2*I+i] - sol[2*I+i-1],0]) 
                            + sol[I+i]*np.min([sol[2*I+i] - sol[2*I+i-1],0]))*2/((x_[i+1] - x_[i-1]) * (sol[2*I+i] - sol[2*I+i-1])) ) / (x_[i+1] - x_[i]) )
            
            # fill diag
            #--- dNP- / dp--------------------------------------------------------------------------------
            if not (sol[2*I+i+1] - sol[2*I+i]) == 0 and (sol[2*I+i] - sol[2*I+i-1]) == 0:

                # case 1
                #--- dNP+ / dp--------------------------------------------------------------------------------
                M[i,2*I+i] = ( Dt*(DC[i+1]*chi1*( sol[i]*np.max([sol[2*I+i+1] - sol[2*I+i],0]) 
                        + sol[i+1]*np.min([sol[2*I+i+1] - sol[2*I+i],0]) )*2*( (x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i])) )/ (x_[i+1] - x_[i]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                M[I+i,2*I+i] = ( Dt*(-DA[i+1]*chi1*( sol[I+i]*np.max([sol[2*I+i+1] - sol[2*I+i],0]) 
                        + sol[I+i+1]*np.min([sol[2*I+i+1] - sol[2*I+i],0]) )*2*( (x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i])) )/ (x_[i+1] - x_[i]) )

            elif not (sol[2*I+i] - sol[2*I+i-1]) == 0 and (sol[2*I+i+1] - sol[2*I+i]) == 0:

                # case 2
                #--- dNP+ / dp--------------------------------------------------------------------------------
                M[i,2*I+i] = ( Dt*(DC[i]*chi1*( sol[i-1]*np.max([sol[2*I+i] - sol[2*I+i-1],0]) 
                        + sol[i]*np.min([sol[2*I+i] - sol[2*I+i-1],0]))*2*( (x_[i+1] - x_[i-1])*(sol[2*I+i] - sol[2*I+i-1])) )/ (x_[i+1] - x_[i]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                M[I+i,2*I+i] = ( Dt*(-DA[i]*chi1*( sol[I+i-1]*np.max([sol[2*I+i] - sol[2*I+i-1],0]) 
                        + sol[I+i]*np.min([sol[2*I+i] - sol[2*I+i-1],0]))*2*( (x_[i+1] - x_[i-1])*(sol[2*I+i] - sol[2*I+i-1])) )/ (x_[i+1] - x_[i]) )

            elif not (sol[2*I+i+1] - sol[2*I+i]) == 0 and not (sol[2*I+i] - sol[2*I+i-1]) == 0:
                
                # case 3
                #--- dNP+ / dp--------------------------------------------------------------------------------
                M[i,2*I+i] = ( Dt*(DC[i+1]*chi1*( sol[i]*np.max([sol[2*I+i+1] - sol[2*I+i],0]) 
                        + sol[i+1]*np.min([sol[2*I+i+1] - sol[2*I+i],0]) )*2*( (x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i]))
                        + DC[i]*chi1*( sol[i-1]*np.max([sol[2*I+i] - sol[2*I+i-1],0]) 
                        + sol[i]*np.min([sol[2*I+i] - sol[2*I+i-1],0]))*2*( (x_[i+1] - x_[i-1])*(sol[2*I+i] - sol[2*I+i-1])) )/ (x_[i+1] - x_[i]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                M[I+i,2*I+i] = ( Dt*(-DA[i+1]*chi1*( sol[I+i]*np.max([sol[2*I+i+1] - sol[2*I+i],0]) 
                        + sol[I+i+1]*np.min([sol[2*I+i+1] - sol[2*I+i],0]) )*2*( (x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i]))
                        - DA[i]*chi1*( sol[I+i-1]*np.max([sol[2*I+i] - sol[2*I+i-1],0]) 
                        + sol[I+i]*np.min([sol[2*I+i] - sol[2*I+i-1],0]))*2*( (x_[i+1] - x_[i-1])*(sol[2*I+i] - sol[2*I+i-1])) )/ (x_[i+1] - x_[i]) )
            
            # fill upper diag
            if not (sol[2*I+i+1] - sol[2*I+i]) == 0:
                
                #--- dNP+ / dp--------------------------------------------------------------------------------
                M[i,2*I+i+1] = ( Dt*(-DC[i+1]*chi1*(sol[i]*np.max([sol[2*I+i+1] - sol[2*I+i],0]) 
                        + sol[i+1]*np.min([sol[2*I+i+1] - sol[2*I+i],0]))*2/((x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i])) ) / (x_[i+1] - x_[i]) )
                
                #--- dNP- / dp--------------------------------------------------------------------------------
                M[I+i,2*I+i+1] = ( Dt*(DA[i+1]*chi1*(sol[I+i]*np.max([sol[2*I+i+1] - sol[2*I+i],0])
                        + sol[I+i+1]*np.min([sol[2*I+i+1] - sol[2*I+i],0]))*2/((x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i])) ) / (x_[i+1] - x_[i]) )

        if i == 0:

            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # dphi0
            M[2*I,2*I] = (-2.0/(x_[2] - x_[0]) - 1.0/(x_[1] - x_[0])) * 1.0/(x_[1] - x_[0])

            # dphi1
            M[2*I,2*I+1] =  2.0/(x_[2] - x_[0]) * 1.0/(x_[1] - x_[0])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # dc0
            M[0,0] = (1 + Dt * ( 2*DC[1]/(x_[2] - x_[0]) -DC[1]*chi1*2* np.max([sol[2*I+1] - sol[2*I],0])/(x_[2] - x_[0]) 
                                - dfAdc ) / (x_[1] - x_[0]) )
        
            # dc1
            M[0,1] = Dt * (-2*DC[1]/(x_[2] - x_[0]) -DC[1]*chi1*2* np.min([sol[2*I+1] - sol[2*I],0])/(x_[2] - x_[0]) ) / (x_[1] - x_[0])

            #--- dNP- / dc--------------------------------------------------------------------------------
            # dc0
            M[I,I] = (1 + Dt * ( 2*DA[1]/(x_[2] - x_[0]) + DA[1]*chi1*2* np.max([sol[2*I+1] - sol[2*I],0])/(x_[2] - x_[0]) ) / (x_[1] - x_[0]) )
        
            # dc1
            M[I,I+1] = Dt * (-2*DA[1]/(x_[2] - x_[0]) + DA[1]*chi1*2* np.min([sol[2*I+1] - sol[2*I],0])/(x_[2] - x_[0]) ) / (x_[1] - x_[0])

            if not (sol[2*I+1] - sol[2*I]) == 0:
                #--- dNP+ / dp--------------------------------------------------------------------------------
                # dp0
                M[0,2*I] = (Dt*( DC[1]*chi1*(sol[0]*np.max([sol[2*I+1] - sol[2*I],0])
                    + sol[1]*np.min([sol[2*I+1] - sol[2*I],0]))*2/((x_[2] - x_[0])*(sol[2*I+1] - sol[2*I]))- dfAdp)/(x_[1] - x_[0]) )

                # dp1
                M[0,2*I+1] = (Dt*(-DC[1]*chi1*(sol[0]*np.max([sol[2*I+1] - sol[2*I],0])
                    + sol[1]*np.min([sol[2*I+1] - sol[2*I],0]))*2/((x_[2] - x_[0])*(sol[2*I+1] - sol[2*I])) )/(x_[1] - x_[0]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                # dp0
                M[I,2*I] = (Dt*(-DA[1]*chi1*(sol[I]*np.max([sol[2*I+1] - sol[2*I],0])
                    + sol[I+1]*np.min([sol[2*I+1] - sol[2*I],0]))*2/((x_[2] - x_[0])*(sol[2*I+1] - sol[2*I])))/(x_[1] - x_[0]) )

                # dp1
                M[I,2*I+1] = (Dt*(DA[1]*chi1*(sol[I]*np.max([sol[2*I+1] - sol[2*I],0])
                    + sol[I+1]*np.min([sol[2*I+1] - sol[2*I],0]))*2/((x_[2] - x_[0])*(sol[2*I+1] - sol[2*I])) )/(x_[1] - x_[0]) )

        if i == I-1:
        
            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # dphiI-2
            M[3*I-1,3*I-2] = 2.0/(x_[I] - x_[I-2]) * 1.0/(x_[I] - x_[I-1])

            # dphiI-1
            M[3*I-1,3*I-1] = (-1.0/(x_[I] - x_[I-1]) - 2.0/(x_[I] - x_[I-2])) * 1.0/(x_[I] - x_[I-1])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # dcI-2
            M[I-1,I-2] = Dt * ( -2*DC[I-1]/(x_[I] - x_[I-2]) + DC[I-1]*chi1*2* np.max([sol[3*I-1] - sol[3*I-2],0])/(x_[I] - x_[I-2]) ) / (x_[I] - x_[I-1])

            # dcI-1
            M[I-1,I-1] = (1+ Dt*( dfCdc + 2*DC[I-1]/(x_[I] - x_[I-2]) 
                          + DC[I-1]*chi1*2* np.min([sol[3*I-1] - sol[3*I-2],0])/(x_[I] - x_[I-2]) )/(x_[I] - x_[I-1]) )

            #--- dNP- / dc--------------------------------------------------------------------------------
            # dcI-2
            M[2*I-1,2*I-2] = Dt * ( -2*DA[I-1]/(x_[I] - x_[I-2]) - DA[I-1]*chi1*2* np.max([sol[3*I-1] - sol[3*I-2],0])/(x_[I] - x_[I-2]) ) / (x_[I] - x_[I-1])

            # dcI-1
            M[2*I-1,2*I-1] = (1+ Dt*( 2*DA[I-1]/(x_[I] - x_[I-2]) 
                          - DA[I-1]*chi1*2* np.min([sol[3*I-1] - sol[3*I-2],0])/(x_[I] - x_[I-2]) )/(x_[I] - x_[I-1]) )

            
            if not (sol[3*I-1] - sol[3*I-2]) == 0:
                #--- dNP+ / dp--------------------------------------------------------------------------------
                # dpI-2
                M[I-1,2*I+I-2] = (Dt*(-DC[I-1]*chi1*( sol[I-2]*np.max([sol[3*I-1] - sol[3*I-2],0])
                    + sol[I-1]*np.min([sol[3*I-1] - sol[3*I-2],0]) )*2/((x_[I] - x_[I-2])*(sol[3*I-1] - sol[3*I-2])))/(x_[I] - x_[I-1]) )

                # dpI-1
                M[I-1,2*I+I-1] = (Dt*(dfCdp + DC[I-1]*chi1*(sol[I-2]*np.max([sol[3*I-1] - sol[3*I-2],0])
                    + sol[I-1]*np.min([sol[3*I-1] - sol[3*I-2],0]))*2/((x_[I] - x_[I-2])*(sol[3*I-1] - sol[3*I-2])) )/(x_[I] - x_[I-1]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                # dpI-2
                M[2*I-1,2*I+I-2] = (Dt*(DA[I-1]*chi1*(sol[2*I-2]*np.max([sol[3*I-1] - sol[3*I-2],0])
                    + sol[2*I-1]*np.min([sol[3*I-1] - sol[3*I-2],0]))*2/((x_[I] - x_[I-2])*(sol[3*I-1] - sol[3*I-2])))/(x_[I] - x_[I-1]) )

                # dpI-1
                M[2*I-1,2*I+I-1] = (Dt*(-DA[I-1]*chi1*(sol[2*I-2]*np.max([sol[3*I-1] - sol[3*I-2],0])
                    + sol[2*I-1]*np.min([sol[3*I-1] - sol[3*I-2],0]))*2/((x_[I] - x_[I-2])*(sol[3*I-1] - sol[3*I-2])) )/(x_[I] - x_[I-1]) )

    return M





#%% Setup

T = 300
epsilon_m = 80 # dielectric constant of water
L =  10 # electrode separation
lengthscalefactor = 1e9 # in nm
c0 = 1 * 1e3 * res._AVOGADRO_CONST / (lengthscalefactor**3) # reference concentration
zC = 1
zA = -1
phi0 = res._BOLTZMANN_CONST * T / res._ELE_CHARGE # thermal energy
D0 = 1e-13 * (lengthscalefactor**2)

# calculation of numerical constants
T0 = L**2 / D0 # reference time
f0 = L*c0/T0 # calculate reference flux density
chi1 = res.calcChi1( phi0, T )
chi2 = res.calcChi2(L, c0, phi0, epsilon_m)*lengthscalefactor

# time axis calculation
# normed t axis, so tmax [s] = tmax x T0
Dt = 0.001
N =  2**15

# computational axis, defines continous values and Dx
I = 2**10 +2
x_ = np.zeros(I+1)
Dx = 1/(I)
for i in range(0,I+1):
    x_[i] = i*Dx

# coordinates of cell centers
x = np.zeros(I)
for i in range(1,I+1):
    x[i-1] = x_[i-1]+(x_[i]-x_[i-1])/2

DC = np.ones(x_.shape)* 1e-11 * (lengthscalefactor**2) / D0
DA = np.ones(x_.shape)* 1e-11 * (lengthscalefactor**2) / D0
epsilon = np.ones(x_.shape)

# reaction rates --> just use blocking electrode
jC0 = 0.0
kC0 = 0.0
jCI = 0.0
kCI = 0.0

# calculate t_ axis, maybe cutted from before
t_ = np.zeros(N)
for j in range(0,N):
    t_[j] = j*Dt

sol = np.zeros([3*x.size,t_.size], dtype = np.float64)
sol[0:I,0] = 1/2
sol[I:2*I,0] = 1/2


#%%
t1 = time.clock()
Jac = calcJac(I, sol[:,0], x_, DC, DA, chi1, chi2, Dt )
Jacinv = np.linalg.inv(Jac)
t2 = time.clock()
print(t2-t1)

#%%
%timeit calcJac(I, sol[:,0], x_, DC, DA, chi1, chi2, Dt )

#%%
%timeit calcJac_new(I, sol[:,0], x_, DC, DA, chi1, chi2, Dt )

#%% Teste np linalg norm oder sum

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 17:51:06 2018

@author: desktop
"""

import numpy as np
import system_tools as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rc('text', usetex=False)
plt.rc('font', family='serif')


#%%
# create x axis
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

T = 300 # temperature
epsilon_m = 80 # dielectric constant of water
L = 10 # electrode separation
lengthscalefactor = 1e9 # in nm
c0 = 1 * 1e3 * st._AVOGADRO_CONST / (lengthscalefactor**3) # reference concentration
phi0 = st._BOLTZMANN_CONST * T / st._ELE_CHARGE # thermal energy

# normed t axis, so tmax [s] = tmax x T0
Dt = 0.0001
N =  2**10

# calculation of numerical constants
D0 = 1e-11 * (lengthscalefactor**2)
T0 = L**2 / D0 # reference time
f0 = L*c0/T0 # calculate reference flux density


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
DC = np.ones(x_.shape) 
DA = np.ones(x_.shape)

# calculate t_ axis
t_ = np.zeros(N)
for j in range(0,N):
	t_[j] = j*Dt

# initialize solution

cc = np.zeros([I,t_.size], dtype = np.float64)
ca = np.zeros([I,t_.size], dtype = np.float64)
phi = np.zeros([I,t_.size], dtype = np.float64)

cc[:,0] = 1.0 / 2.0
ca[:,0] = 1.0 / 2.0

# permittivity and diffusion constants
epsilon = np.ones(x_.shape)
DC = np.ones(x_.shape) 
DA = np.ones(x_.shape)

# reaction rates --> just use blocking electrode
kC_vec = np.array([1.0 * 1e-1], dtype = np.float64)
kA_vec = np.array([1.0 * 1e-1] , dtype = np.float64)
foxA_vec = np.array([0.5 * 1e-1] , dtype = np.float64)
foxC_vec = np.array([0.5 * 1e-1] , dtype = np.float64)

voltage = 10.0

#%%

def createMat( I, x_, Dt, D ):
	# create system matrix
	M = np.zeros([I,I])
	
	# fill inner elements
	for i in range(1,I-1):
		
		M[i,i+1] = -2 * Dt * D[i+1] / ( (x_[i+1] - x_[i]) * (x_[i+2] - x_[i]) )
		M[i,i] = ( 1 + 2 * Dt * D[i+1] / ( (x_[i+1] - x_[i]) * (x_[i+2] - x_[i]) ) 
				+ 2 * Dt * D[i] / ( (x_[i+1] - x_[i]) * (x_[i+1] - x_[i-1]) ) )
		M[i,i-1] = -2 * Dt * D[i] / ( (x_[i+1] - x_[i]) * (x_[i+1] - x_[i-1]) )
		
	# boundary elements
	M[0,0] = 1 + 2 * Dt * D[1] / ( (x_[1] - x_[0]) * (x_[2] - x_[0]) )
	M[0,1] = -2 * Dt * D[1] / ( (x_[1] - x_[0]) * (x_[2] - x_[0]) )
	M[I-1,I-2] = -2 * Dt * D[I-1] / ( (x_[I] - x_[I-1]) * (x_[I] - x_[I-2]) )
	M[I-1,I-1] = 1 + 2 * Dt * D[I-1] / ( (x_[I] - x_[I-1]) * (x_[I] - x_[I-2]) )
	
	return M


def calcRHS(I, c, phi, z, chi1, x_, Dt, D ):
	rhs = np.zeros(I, dtype = np.float64)
		
	# boundary conditions model 0
	fA = 0.0
	fC = 0.0
	# Anode, upflux only, substitute downflux with boundary condition
	# calculate upwinding velocities
	velup = -2 * chi1 * z * D[1] * (phi[1] - phi[0]) / (x_[2] - x_[0])
	    
	# calc upwinding
	if velup >=0.0:
		upflux = c[0] * velup 
	else:
		upflux = c[1] * velup
	
	rhs[0] = - Dt * ( upflux - fA ) / (x_[1] - x_[0]) + c[0]
	
	# inner points
	for i in range(1,I-1):
		
		# calculate upwinding velocities
		veldown = -2 * chi1 * z * D[i] * (phi[i] - phi[i-1]) / (x_[i+1] - x_[i-1])
		velup = -2 * chi1 * z * D[i+1] * (phi[i+1] - phi[i]) / (x_[i+2] - x_[i])
		
		if  veldown >= 0.0:
			downflux = c[i-1] * veldown
		else:
			downflux = c[i] * veldown
	            
		if  velup >= 0.0:
			upflux = c[i] * velup
		else:
			upflux = c[i+1] * velup
			
		rhs[i] = - Dt * (upflux - downflux) / (x_[i+1] - x_[i]) + c[i]
		
	# calculate upwinding velocities
	veldown = -2 * chi1 * z * D[I-1] * (phi[I-1] - phi[I-2]) / (x_[I] - x_[I-2])
	    
	if veldown >=0.0:
		downflux = c[I-2] * veldown
	else:
		downflux = c[I-1] * veldown
	
	rhs[I-1] = -Dt * ( fC - downflux ) / (x_[I] - x_[I-1]) + c[I-1]
	
	return rhs

# Potential
def solvePoisson( I, x_, cc, ca, chi2, phiC, epsilon):
	N = np.zeros([I,I], dtype = np.float64)
	
	# anode boundary
	N[0,0] = -( 2 * epsilon[1] /  (x_[2] - x_[0])  + epsilon[0] / (x_[1] - x_[0]) ) / (x_[1] - x_[0])
	N[0,1] = 2 * epsilon[1] / ( (x_[1] - x_[0]) * (x_[2] - x_[0]))
	
	# inner points
	for i in range(0,I-1):
		
		N[i,i-1] = 2 * epsilon[i] / ( (x_[i+1] - x_[i]) * (x_[i+1] - x_[i-1]) )
		
		N[i,i] = -2 * ( epsilon[i+1] / (x_[i+2] - x_[i]) + epsilon[i] / (x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i])
		
		N[i,i+1] = 2 * epsilon[i+1] / ( (x_[i+1] - x_[i]) * (x_[i+2] - x_[i]) )
	
	# catode boundary
	N[I-1,I-2] = 2 * epsilon[I-1] / ( (x_[I] - x_[I-1]) * (x_[I] - x_[I-2]) )
	
	N[I-1,I-1] = -( epsilon[I] / (x_[I] - x_[I-1]) + 2 * epsilon[I-1] / (x_[I] - x_[I-2]) ) / (x_[I] - x_[I-1])
	
	
	rhs_phi = np.zeros(I, dtype = np.float64)
	rhs_phi = -chi2 * (cc - ca)
	rhs_phi[I-1] = -chi2 * (cc[I-1] - ca[I-1]) - epsilon[I] * phiC / ( (x_[I] - x_[I-1]) * (x_[I] - x_[I-1]) )
	
	phi = np.linalg.solve(N, rhs_phi)
	
	return phi
	
	
#%% Testcase Poisson Equation
	
	

#%% Solution Solver loop


phiC = np.ones(t_.shape[0], dtype = np.float64) * voltage * 1e-3 / phi0

Mc = createMat( I, x_, Dt, DC )
Ma = createMat( I, x_, Dt, DA )

# solution loop
for j in range(0,t_.size-1):

	if j <= 100:
		print("Time Step: ", j)
	elif j > 100 and np.mod(j,100) == 0:
		print("Time Step: ", j)

	# create rhs from j
	rhsc = calcRHS(I, cc[:,j], phi[:,j], 1, chi1, x_, Dt, DC )
	rhsa = calcRHS(I, ca[:,j], phi[:,j], -1, chi1, x_, Dt, DA )
	
	# solve for cations / anions for j+1
	cc[:,j+1] = np.linalg.solve(Mc, rhsc)
	ca[:,j+1] = np.linalg.solve(Ma, rhsa)
	
	# solve for potential for j+1
	phi[:,j+1] = solvePoisson( I, x_, cc[:,j+1], ca[:,j+1], chi2, phiC[j+1], epsilon)


#        if np.linalg.norm(sol[0:I,j] - sol[0:I,j-1],ord=2) < 1e-8:
#            print("Steady State after ",j," Time Steps")
#            break
#	
#    # cut vectors
#    if j < t_.shape[0]:
#        t_ = np.delete(t_, np.s_[j:], axis = 0)
#        sol = np.delete(sol, np.s_[j:], axis = 1)
#        phiC = np.delete(phiC, np.s_[j:], axis = 0)

#    t2 = time.clock()
#    print("Simulation Runtime Old Implementation: ", t2-t1)


#%% Plotting
saveflag = 0

index = [10]

# plot cations in subplot 1
fig, axes = plt.subplots(3, 1, sharex='all')# gridspec_kw = {'wspace':0.1, 'hspace':0.4})

ax = axes[0]
for i in range(0,index.__len__()):

    ax.plot(x * L, cc[:,index[i]] * c0 * 1e24 / st._AVOGADRO_CONST,)# color = colorlist[i], ls = lslist[i], label = labels[i])


# ticks formatting
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# grid
ax.grid(b = True, which = "major", axis = "both")

# axis labels
ax.set_ylabel(r" Cations [M]",fontsize = 9,  fontweight = "bold")

# get legend
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels )


# plot anions in subplot 2
ax = axes[1]
for i in range(0,index.__len__()):

    ax.plot(x * L, ca[:,index[i]] * c0 * 1e24 / st._AVOGADRO_CONST,)#  color = colorlist[i], ls = lslist[i], label = labels[i])


# ticks formatting
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# grid
ax.grid(b = True, which = "major", axis = "both")

# axis labels
#ax.set_xlabel(r"x [nm]")
ax.set_ylabel(r" Anions [M]",fontsize = 9,  fontweight = "bold")

# get legend
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels )


# plot potential in subplot 3
ax = axes[2]
for i in range(0,index.__len__()):
# - phi[int(I/2),index[i]]
    ax.plot(x * L, ( phi[:,index[i]] )* phi0 * 1e3,)#  color = colorlist[i], ls = lslist[i], label = labels[i])

# ticks formatting
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# grid
ax.grid(b = True, which = "major", axis = "both")

# axis labels
ax.set_xlabel(r"x [nm]", fontweight = "bold")
ax.set_ylabel(r"Potential [mV]",fontsize = 9,  fontweight = "bold")

# get legend
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels )

# adjust subplits
#plt.subplots_adjust( wspace=0, hspace=0)

# tight layout
plt.tight_layout() #pad=5, w_pad=0.5, h_pad=0.5)

if saveflag == 1:
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(figs_path, "all_sols_anodic_flux." +fmt ), format = fmt, dpi = 400)

plt.show()


# %%
cc_int = np.sum(cc * Dx, axis = 0 )
ca_int = np.sum(cc * Dx, axis = 0 )

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(t_[0:700], cc_int[0:700], color = "blue")
ax.plot(t_[0:700], ca_int[0:700], color = "red")

plt.show()


#%% Diffusion / Impedance Test Script
machine = "work"

import os 

print("Current working directory :", os.getcwd())

# import
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rc('text', usetex=False)
plt.rc('font', family='serif')

from solver import PDEsolver, createAxis
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
N =  2**14

# calculation of numerical constants
D0 = 1e-11 * (lengthscalefactor**2)
T0 = L**2 / D0 # reference time
f0 = L*c0/T0 # calculate reference flux density

# reaction rates --> just use blocking electrode
kC_vec = np.array([1.0 * 1e-1], dtype = np.float64)
kA_vec = np.array([1.0 * 1e-1] , dtype = np.float64)
foxA_vec = np.array([0.5 * 1e-1] , dtype = np.float64)
foxC_vec = np.array([0.5 * 1e-1] , dtype = np.float64)

# calculation of numerical constants
chi1 = st.calcChi1( phi0, T )
chi2 = st.calcChi2(L, c0, phi0, epsilon_m)*lengthscalefactor
    
# computational axis, defines continous values and Dx
I = 2**9 +2
x_, x = createAxis( I )

# permittivity and diffusion constants
epsilon = np.ones(x_.shape)
DC = np.ones(x_.shape) 
DA = np.ones(x_.shape)

# voltage vector in [mV]
voltage = np.array([10.0], dtype = np.float64)

# output list
current_list = []
sol_list = []

#%% Simulation loop

# initiate intial condition
sol_initial = np.zeros(3*I, dtype = np.float64)
sol_initial[0:2*I] = 1/2


for iterI in range(0,kA_vec.size):
    
    print("testnummer:",iterI)
    
    # voltage loop
    for iterK in range(0,voltage.size):
        
        # calculate t_ axis
        t_ = np.zeros(N)
        for j in range(0,N):
            t_[j] = j*Dt

        # boundary condition values
        kA = kA_vec[iterI]
        kC = kC_vec[iterI]
        foxA = foxA_vec[iterI]
        foxC = foxC_vec[iterI]

        # calculate t_ axis, maybe cutted from before
        phiC = np.ones(t_.shape, dtype = np.float64) * voltage[iterK] * 1e-3 / phi0
    
        # call pde solver
        sol = PDEsolver( I, x_, t_, sol_initial, phiC, DC, DA, epsilon, chi1, chi2, Dt, kA, kC, foxA, foxC)

        # calculate result values 
        current = np.zeros([2,sol.shape[1]], dtype = np.float64)
        for j in range(0,sol.shape[1]):
            current[0,j] = foxA - kA * sol[0,j] - ( sol[2*I,j] - sol[2*I,j-1] ) /((x_[1]-x_[0])*Dt*chi2)
            current[1,j] = kC * sol[I-1,j] - foxC - ( phiC[j] - sol[3*I-1,j] - phiC[j-1] + sol[3*I-1,j-1] ) /((x_[I] - x_[I-1])*Dt*chi2)

        current_list.append(current)
        sol_list.append(sol[:,-1])

        del current, t_, phiC

#%% save solution

#np.save(os.path.join(input_files_path,"impedance_input_1.npy"), sol[:,-1])



#%% plot all currents

#colorlist = ["red", "red","blue", "blue"]
#lslist = ["-", "--", "-", "--"]
saveflag = 0
if saveflag == 1:
    figs_path = r"M:\QMR\Abteilungsprojekte\FG\008_masterthesis\implementations\model_1\solutions\sweep1"
    figname = r"sweep1_currents"

fig = plt.figure()
ax = fig.add_subplot(1,1,1)


for i in range(14,current_list.__len__()):

    # calculate t_ axis
    t_ = np.zeros(current_list[i][0,:].size)
    for j in range(0,current_list[i][0,:].size):
        t_[j] = j*Dt

    ax.plot(t_[2:] * T0, current_list[i][0,2:] * f0 * st._ELE_CHARGE * 1e14 * 1e3, lw = 2 , label = r"Catode Voltage " + str(voltage[i]) + r"$\left[mV\right]$") # label = labels[i], color = colorlist[i], lw = 2,ls = lslist[i])

# ticks formatting
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%1.0e"))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
#ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))

# grid
ax.grid(b = True, which = "major", axis = "both")
ax.grid(b = True, which = "minor", axis = "both")

# axis labels
ax.set_xlabel(r"Time $\bf{[s]}$",fontsize = 10,  fontweight = "bold")
ax.set_ylabel(r"Anode Current Density $ \mathbf{\left[mA cm^{-2}\right]}$",fontsize = 10,  fontweight = "bold")

# get legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# tight layout
fig.tight_layout()

if saveflag == 1:
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(figs_path, figname + "." +fmt ), format = fmt, dpi = 300)

plt.show()

#%% U - I - Plot
saveflag = 1
if saveflag == 1:
    figs_path = r"M:\QMR\Abteilungsprojekte\FG\008_masterthesis\implementations\model_1\solutions\sweep1"
    figname = r"sweep1_ss_currents_all"

ss_out = np.zeros([current_list.__len__(), 3], dtype = np.float64)
# extract steady state currents
for i in range(0,current_list.__len__()):
    ss_out[i,1] = current_list[i][0,-1]
    ss_out[i,2] = current_list[i][1,-1]

ss_out[0:voltage.size,0] = voltage
ss_out[voltage.size:2*voltage.size,0] = voltage
ss_out[2*voltage.size:3*voltage.size,0] = voltage

filename = os.path.join(r"M:\QMR\Abteilungsprojekte\FG\008_masterthesis\implementations\model_1\solutions\sweep1","ss_currents.npy")
np.save(filename,ss_out)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for i in range(0, voltage.size):
    ax.plot(ss_out[i,0], ss_out[i,1] * f0 * st._ELE_CHARGE * 1e14 * 1e3 , ls = "None", marker = ".", markersize = 8, color = "blue")

for i in range(voltage.size, 2*voltage.size):
    ax.plot(ss_out[i,0], ss_out[i,1] * f0 * st._ELE_CHARGE * 1e14 * 1e3 , ls = "None", marker = ".", markersize = 8, color = "red")

for i in range(2*voltage.size, 3*voltage.size):
    ax.plot(ss_out[i,0], ss_out[i,1] * f0 * st._ELE_CHARGE * 1e14 * 1e3 , ls = "None", marker = ".", markersize = 8, color = "green")

# legend
ax.plot([], [], ls = "None", marker  = ".", markersize = 8, color = "red", label = "Rate Coefficient Factor ${0.1}$")
ax.plot([], [], ls = "None", marker  = ".", markersize = 8, color = "blue", label = "Rate Coefficient Factor ${1.0}$")
ax.plot([], [], ls = "None", marker  = ".", markersize = 8, color = "green", label = "Rate Coefficient Factor ${10.0}$")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# labels
ax.set_xlabel(r"Voltage $\mathbf{\left[mV\right]}$", fontsize = 10,  fontweight = "bold")
ax.set_ylabel(r"Anode Current Density $\mathbf{\left[mA cm^{-2}\right]}$",fontsize = 10,  fontweight = "bold")

# ticks formatting
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# limits
#ax.set_ylim([-50,50])

# grid
ax.grid(b = True, which = "major", axis = "both")

if saveflag == 1:
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(figs_path, figname + "." +fmt ), format = fmt, dpi = 300)

plt.show()

#%%
#labels = ["Anodic Flux Factor 1", "Anodic Flux Factor 10", "Anodic Flux Factor 100","Anodic Flux Factor 1000"]
#colorlist = ["red", "red","blue", "blue"]
#lslist = ["-", "--", "-", "--"]

saveflag = 0

# indexes of plotlist
index_b = sol_list.__len__() - 3
index_e = sol_list.__len__()

# plot cations in subplot 1
fig, axes = plt.subplots(3, 1, sharex='all')# gridspec_kw = {'wspace':0.1, 'hspace':0.4})

ax = axes[0]
for i in range(index_b,index_e):

    ax.plot(x * L, sol_list[i][0:I] * c0 * 1e24 / st._AVOGADRO_CONST,)# color = colorlist[i], ls = lslist[i], label = labels[i])


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
for i in range(index_b,index_e):

    ax.plot(x * L, sol_list[i][I:2*I] * c0 * 1e24 / st._AVOGADRO_CONST,)#  color = colorlist[i], ls = lslist[i], label = labels[i])


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
for i in range(index_b,index_e):

    ax.plot(x * L, ( sol_list[i][2*I:3*I] - sol_list[i][2*I+int(I/2)])* phi0 * 1e3,)#  color = colorlist[i], ls = lslist[i], label = labels[i])

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


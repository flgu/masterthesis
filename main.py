#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:34:13 2018

@author: desktop
"""

# import libraries
from functions.solver import ImpedanceSolver_m0
from functions.read_setup import setup
from functions.impedance_functions import createImpedanceVoltage
import numpy as np



if __name__ == "__main__":

    testname = r"solutions/imp11"

    # import setup
    path_2_setupfile = r"setup.txt"


    stp = setup( path_2_setupfile )

    # create voltage
    phiC = createImpedanceVoltage(stp.N, stp.Dt, stp.T0, stp.phi0, U_offset = 0, num = 80)

    np.save(testname + r"input_volt.npy",phiC)

    #phiC = np.ones(stp.N, dtype = np.float64) * 0.01 / stp.phi0

    # initial condition
    sol_initial = np.zeros([3 * stp.I], dtype = np.float64)
    sol_initial[0:2*stp.I] = 1.0 / 2.0

    # call solver
    current = ImpedanceSolver_m0( stp.I,
                                 stp.L,
                                 stp.c0,
                                 stp.epsilon_m,
                                 stp.phi0,
                                 stp.T,
                                 stp.N,
                                 stp.Dt,
                                 stp.M,
                                 sol_initial,
                                 phiC,
                                 stp.DC,
                                 stp.DA,
                                 stp.epsilon,
                                 stp.model,
                                 stp.kA,
                                 stp.foxA,
                                 stp.kC,
                                 stp.foxC )

    # dont save because its a test file
    np.save(testname + "current" + r".npy", current)

#%%
import matplotlib.pyplot as plt

time = np.zeros(stp.N)

for j in range(0,stp.N):

    time[j] = j * stp.Dt

fig  = plt.figure(dpi = 150)

ax = fig.add_subplot(1,1,1)

ax.plot(time, phiC)

ax.plot( [8400* stp.Dt, 8400* stp.Dt], [-0.2,0.2] )

plt.show()







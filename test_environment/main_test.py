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
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # import setup
    path_2_setupfile = r"test_setup.txt"


    stp = setup( path_2_setupfile )
    stp.Dt = 1e-5
    M = 1e0

    stp.T0 = M * stp.L ** 2 / stp.D0

    # create voltage
    phiC = createImpedanceVoltage(stp.N, stp.Dt, stp.T0, stp.phi0, U_offset = 0, num = 40)
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
                                 M,
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

#%% plotting
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(current[0,2:])
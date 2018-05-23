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

    # import setup
    #path_2_setupfile = r"M:\QMR\Abteilungsprojekte\FG\local\masterthesis\setup.txt"
    path_2_setupfile = r"setup.txt"
    #path_2_solution = r"M:\QMR\Abteilungsprojekte\FG\local\masterthesis\solutions\impedance_test_4.npy"
    path_2_solution = r"solutions/impedance_test_9..npy"
    
    stp = setup( path_2_setupfile )

    # create voltage
    phiC = createImpedanceVoltage(stp.N, stp.Dt, stp.T0, stp.phi0, U_offset = 0, num = 40)
    
    # initial condition
    sol_initial = np.zeros([3 * stp.I], dtype = np.float64)
    sol_initial[0:2*stp.I] = 1.0 / 2.0
    
    # call solver
    current = ImpedanceSolver_m0( stp.I, stp.L, stp.c0, stp.epsilon_m, stp.phi0, stp.T, stp.N, stp.Dt, sol_initial, phiC, stp.DC, stp.DA, stp.epsilon)
    
    # save output    
    np.save(path_2_solution, current)
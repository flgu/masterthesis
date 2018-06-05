import numpy as np

import multiprocessing as mp

from time import clock
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import os

from pre_fun import *


# Teste Genauigkeit der Steady State Analysis
voltage_list = [-40e-3, -30e-3, -20e-3, -10e-3, -7.5e-3, -5e-3, -2.5e-3, -1e-3, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 10e-3, 20e-3, 30e-3, 40e-3]

for i in range( 0,voltage_list.__len__() ):
    print(i)
    stp_name = '20180605_' + 'test_' + str(i) + '_model_1_k1'
    stp = Setup(  I = 500,
                    N = 100000,
                    Dt = 1e-3,
                    T = 300,
                    L = 10,
                    lengthscale = 1e9,
                    c0_in = 1.0,
                    DA = 1.0,
                    DC = 1.0,
                    D0_in = 1e-10,
                    epsilon = 1.0,
                    epsilon_m = 80.0,
                    testname = stp_name,
                    model = 1,
                   sim_method = "single_const",
                   kA = 1e-3,
                   kC = 1e-3,
                   foxA = 0.5e-3,
                       foxC = 0.5e-3, 
                    steady_state_tol = 1e-7,)

    # set pertubation voltage
    stp.set_input_voltage_data( voltage_list[i] )

    # call solver
    stp.solver()

for i in range( 0,voltage_list.__len__() ):
    print(i)
    stp_name = '20180604_' + 'test_Dt_' + str(i) + '_model_1_k2'
    stp = Setup(  I = 500,
                    N = 60000,
                    Dt = 1e-3,
                    T = 300,
                    L = 10,
                    lengthscale = 1e9,
                    c0_in = 1.0,
                    DA = 1.0,
                    DC = 1.0,
                    D0_in = 1e-10,
                    epsilon = 1.0,
                    epsilon_m = 80.0,
                    testname = stp_name,
                    model = 1,
                   sim_method = "single_const",
                   kA = 1e-2,
                   kC = 1e-2,
                   foxA = 0.5e-2,
                       foxC = 0.5e-2, 
                    steady_state_tol = 1e-8,)

    # set pertubation voltage
    stp.set_input_voltage_data( voltage_list[i] )

    # call solver
    stp.solver()

for i in range( 0,voltage_list.__len__() ):
    print(i)
    stp_name = '20180604_' + 'test_Dt_' + str(i) + '_model_1_k3'
    stp = Setup(  I = 500,
                    N = 60000,
                    Dt = 1e-3,
                    T = 300,
                    L = 10,
                    lengthscale = 1e9,
                    c0_in = 1.0,
                    DA = 1.0,
                    DC = 1.0,
                    D0_in = 1e-10,
                    epsilon = 1.0,
                    epsilon_m = 80.0,
                    testname = stp_name,
                    model = 1,
                   sim_method = "single_const",
                   kA = 1e-1,
                   kC = 1e-1,
                   foxA = 0.5e-1,
                       foxC = 0.5e-1, 
                    steady_state_tol = 1e-8,)

    # set pertubation voltage
    stp.set_input_voltage_data( voltage_list[i] )

    # call solver
    stp.solver()
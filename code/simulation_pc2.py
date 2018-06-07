import numpy as np

import multiprocessing as mp

from time import clock
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import os

from pre_fun import *




def caller( setup ):

  setup.solver()
	

if __name__ == "__main__":

    voltage_list = [-40e-3, -30e-3, -20e-3, -15e-3, -10e-3 , -5e-3, -1e-3,
     1e-3, 5e-3, 10e-3, 15e-3, 20e-3, 30e-3, 40e-3 ]

    setup_list = []

    # create setups
    for i in range(0, voltage_list.__len__()):

        name = '20180607_model_1_sweep_test1_' + str(i)

        stp = Setup(  I = 500,
                    I_middle = 0,
                    N = 2 ** 16,
                    Dt = 1e-4,
                    T = 300,
                    L = 10,
                    lengthscale = 1e9,
                    c0_in = 1.0,
                    DA = 1.0,
                    DC = 1.0,
                    D0_in = 1e-10,
                    epsilon = 1.0,
                    epsilon_m = 80.0,
                    testname = name,
                    model = 1,
                    sim_method = "single_const",
                    kA = 1.0,
                    kC = 1.0,
                    E0_A = 0.0,
                    E0_C = 0.0,
                    cA = 0.5,
                    cC = 0.5,
                    alpha = 0.0,
                    steady_state_tol = 1e-7,)

        stp.set_input_voltage_data( voltage_list[i] )

        setup_list.append( stp )

    print("Setup 1 created, len:", setup_list.__len__())

    for i in range(0, voltage_list.__len__()):

        name = '20180607_model_1_sweep_test2_' + str(i)

        stp = Setup(  I = 500,
                    I_middle = 0,
                    N = 2 ** 16,
                    Dt = 1e-4,
                    T = 300,
                    L = 10,
                    lengthscale = 1e9,
                    c0_in = 1.0,
                    DA = 1.0,
                    DC = 1.0,
                    D0_in = 1e-10,
                    epsilon = 1.0,
                    epsilon_m = 80.0,
                    testname = name,
                    model = 2,
                    sim_method = "single_const",
                    kA = 1.0e-1,
                    kC = 1.0e-1,
                    E0_A = 0.0,
                    E0_C = 0.0,
                    cA = 0.5,
                    cC = 0.5,
                    alpha = 0.0,
                    steady_state_tol = 1e-7,)

        stp.set_input_voltage_data( voltage_list[i] )

        setup_list.append( stp )

    print("Setup 2 created, len:", setup_list.__len__())

    for i in range(0, voltage_list.__len__()):

        name = '20180607_model_1_sweep_test3_' + str(i)

        stp = Setup(  I = 500,
                    I_middle = 0,
                    N = 2 ** 16,
                    Dt = 1e-4,
                    T = 300,
                    L = 10,
                    lengthscale = 1e9,
                    c0_in = 1.0,
                    DA = 1.0,
                    DC = 1.0,
                    D0_in = 1e-10,
                    epsilon = 1.0,
                    epsilon_m = 80.0,
                    testname = name,
                    model = 2,
                    sim_method = "single_const",
                    kA = 1.0e-2,
                    kC = 1.0e-2,
                    E0_A = 0.0,
                    E0_C = 0.0,
                    cA = 0.5,
                    cC = 0.5,
                    alpha = 0.0,
                    steady_state_tol = 1e-7,)

        stp.set_input_voltage_data( voltage_list[i] )

        setup_list.append( stp )

    print("Setup 3 created, len:", setup_list.__len__())

    for i in range(0, voltage_list.__len__()):

        name = '20180607_model_1_sweep_test4_' + str(i)

        stp = Setup(  I = 500,
                    I_middle = 0,
                    N = 2 ** 16,
                    Dt = 1e-4,
                    T = 300,
                    L = 10,
                    lengthscale = 1e9,
                    c0_in = 1.0,
                    DA = 1.0,
                    DC = 1.0,
                    D0_in = 1e-10,
                    epsilon = 1.0,
                    epsilon_m = 80.0,
                    testname = name,
                    model = 2,
                    sim_method = "single_const",
                    kA = 1.0e-3,
                    kC = 1.0e-3,
                    E0_A = 0.0,
                    E0_C = 0.0,
                    cA = 0.5,
                    cC = 0.5,
                    alpha = 0.0,
                    steady_state_tol = 1e-7,)

        stp.set_input_voltage_data( voltage_list[i] )

        setup_list.append( stp )

    print("Setup 4 created, len:", setup_list.__len__())

    for i in range(0, voltage_list.__len__()):

        name = '20180607_model_1_sweep_test5_' + str(i)

        stp = Setup(  I = 500,
                    I_middle = 0,
                    N = 2 ** 16,
                    Dt = 1e-4,
                    T = 300,
                    L = 10,
                    lengthscale = 1e9,
                    c0_in = 1.0,
                    DA = 1.0,
                    DC = 1.0,
                    D0_in = 1e-10,
                    epsilon = 1.0,
                    epsilon_m = 80.0,
                    testname = name,
                    model = 2,
                    sim_method = "single_const",
                    kA = 1.0e-4,
                    kC = 1.0e-4,
                    E0_A = 0.0,
                    E0_C = 0.0,
                    cA = 0.5,
                    cC = 0.5,
                    alpha = 0.0,
                    steady_state_tol = 1e-7,)

        stp.set_input_voltage_data( voltage_list[i] )

        setup_list.append( stp )

    print("Setup 5 created, len:", setup_list.__len__())

    # Start simulation
    pool = mp.Pool(processes = 4)

    t1 = clock()

    pool.map(caller, setup_list )

    t2 = clock()
    print("Overall Simulation Time [s]: ", (t2 - t1)  )
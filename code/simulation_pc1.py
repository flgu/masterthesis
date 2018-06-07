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

    k_list = [1e0, 1e-2, 1e-4]

    setup_list = []

    # create setups
    for i in range(0, k_list.__len__()):

        name = '20180607_model_1_imp_test1_' + str(i)

        stp = Setup(  I = 500,
                    I_middle = 0,
                    N = 2 ** 19,
                    Dt = 1e-4,
                    T = 300,
                    L = 40,
                    lengthscale = 1e9,
                    c0_in = 1.0,
                    DA = 1.0,
                    DC = 1.0,
                    D0_in = 1e-10,
                    epsilon = 1.0,
                    epsilon_m = 80.0,
                    testname = name,
                    model = 1,
                    sim_method = "c_only_imp",
                    kA = k_list[i],
                    kC = k_list[i],
                    foxA = 0.0,
                    foxC = 0.0,
                    E0_A = 0.0,
                    E0_C = 0.0,
                    cA = 0.5,
                    cC = 0.5,
                    alpha = 0.0,
                    steady_state_tol = 1e-7,)

        stp.set_input_voltage_data( voltage_list[i] )

        setup_list.append( stp )

    

    # Start simulation
    pool = mp.Pool(processes = 4)

    t1 = clock()

    pool.map(caller, setup_list )

    t2 = clock()
    print("Overall Simulation Time [s]: ", (t2 - t1)  )
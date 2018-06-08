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

    k_list = [1e-4, 1e-6]

    setup_list = []

    # create setups
    for i in range(0, k_list.__len__()):

        name = '20180608_model_1_imp_0_' + str(i)

        stp = Setup(  I = 500,
                    N = 2 ** 21,
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
                    kAox = k_list[i],
                    kAred = k_list[i],
                    kCox = k_list[i],
                    kCred = k_list[i],
                    cA = 0.5,
                    cC = 0.5,
                    alpha = 0.0,
                    steady_state_tol = 1e-7,)

        setup_list.append( stp )

    f_list = [2, 4, 6, 8]    

    for i in range(0, f_list.__len__()):

        name = '20180608_model_2_imp_1_' + str(i)

        stp = Setup(  I = 500,
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
                    kAox = f_list[i] * 1e-2,
                    kAred = 1e-2,
                    kCox = 1e-2,
                    kCred = f_list[i] * 1e-2,
                    cA = 0.5,
                    cC = 0.5,
                    alpha = 0.0,
                    steady_state_tol = 1e-7,)

        setup_list.append( stp )


    print("Len Setup_list", setup_list.__len__() )

    # Start simulation
    pool = mp.Pool(processes = 4)

    t1 = clock()

    pool.map(caller, setup_list )

    t2 = clock()
    print("Overall Simulation Time [s]: ", (t2 - t1)  )
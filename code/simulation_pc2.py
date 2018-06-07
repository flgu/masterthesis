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
    voltage_list = [-30e-3, -20e-3, -15e-3, -10e-3 ,-7.5e-3 -5e-3, -2.5e-3, -1e-3, -0.5e-3,
                0.5e-3, 1e-3, 2.5e-3, 5e-3, 10e-3, 15e-3, 20e-3, 30e-3 ]
    setup_list = []

    # create setups
    for i in range(0, voltage_list.__len__()):

      name = '20180607_model_2_sweep_2_' + str(i)

      stp = Setup(  I = 500,
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
                       foxA = 0.5e-1,
                           foxC = 0.5e-1,
                        E0_A = 0.0,
                        E0_C = 0.0,
                        alpha = 0.5,
                        steady_state_tol = 1e-7,)

      stp.set_input_voltage_data( voltage_list[i] )

      setup_list.append( stp )



    pool = mp.Pool(processes = 2)


    t1 = clock()

   # for i in range(1,setup_list.__len__()):
    #    print('start sim', i)
    pool.map(caller, setup_list )

    t2 = clock()
    print("Overall Simulation Time [s]: ", (t2 - t1)  )
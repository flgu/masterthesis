import numpy as np

import multiprocessing as mp

from time import clock
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import os

from pre_fun import *


    
stp1 = Setup(  I = 500,
                N = 2 ** 21,
                Dt = 1e-4,
                T = 300,
                L = 10,
                lengthscale = 1e9,
                c0_in = 1.0,
                DA = 10.0,
                DC = 10.0,
                D0_in = 1e-11,
                epsilon = 1.0,
                epsilon_m = 80.0,
                testname = '20180605_model_1_impedance_1',
                model = 1,
               sim_method = "c_only_imp",
               kA = 1e-2,
               kC = 1e-2,
               foxA = 0.5e-2,
                   foxC = 0.5e-2, 
                steady_state_tol = 5e-7,)

stp2 = Setup(  I = 500,
                N = 2 ** 21,
                Dt = 1e-4,
                T = 300,
                L = 10,
                lengthscale = 1e9,
                c0_in = 1.0,
                DA = 10.0,
                DC = 10.0,
                D0_in = 1e-11,
                epsilon = 1.0,
                epsilon_m = 80.0,
                testname = '20180605_model_1_impedance_2',
                model = 1,
               sim_method = "c_only_imp",
               kA = 1e-1,
               kC = 1e-1,
               foxA = 0.5e-1,
                   foxC = 0.5e-1, 
                steady_state_tol = 5e-7,)

stp3 = Setup(  I = 500,
                N = 2 ** 21,
                Dt = 1e-4,
                T = 300,
                L = 10,
                lengthscale = 1e9,
                c0_in = 1.0,
                DA = 10.0,
                DC = 10.0,
                D0_in = 1e-11,
                epsilon = 1.0,
                epsilon_m = 80.0,
                testname = '20180605_model_1_impedance_3',
                model = 1,
               sim_method = "c_only_imp",
               kA = 1e-3,
               kC = 1e-3,
               foxA = 0.5e-3,
                   foxC = 0.5e-3, 
                steady_state_tol = 5e-7,)



def caller( setup ):
	setup.solver()
	

if __name__ == "__main__":

    t1 = clock()

	# start multiprocessing
    p1 = mp.Process( target = caller, args = (stp1, ) )
    p2 = mp.Process( target = caller, args = (stp2, ) )
    p3 = mp.Process( target = caller, args = (stp3, ) )
    #p4 = mp.Process( target = solver, args = ( stp4, ))


	# start subprocesses
    p1.start()
    p2.start()
    p3.start()
    #p4.start()


    p1.join()
    p2.join()
    p3.join()
    #p4.join()

    t2 = clock()
    print("Overall Simulation Time [s]: ", (t2 - t1)  )
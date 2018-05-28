import numpy as np

import multiprocessing as mp

from time import clock
from scipy.optimize import newton_krylov
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import os

from solver_tools_new import residual as res, constJac
from pre_fun import Setup, calcAxis

def solver( setup, reaction_bc, phiC ):
    
    I = setup.I
    N = setup.N
    M = 1.0
        
    # initial condition
    sol_initial = np.zeros([3*I], dtype = np.float64)
    sol_initial[0:2*I] = 1.0 / 2.0
    
    # reactin boundary conditions
    kA = reaction_bc["kA"]
    kC = reaction_bc["kC"]
    foxA = reaction_bc["foxA"]
    foxC = reaction_bc["foxC"]
    
    # axis
    Dx, centers = calcAxis( I )
    
    current = np.zeros([2, N], dtype = np.float64)
    epsilon_vec = np.ones(I + 1, dtype = np.float64) * setup.epsilon
    DC_vec = np.ones(I + 1, dtype = np.float64) * setup.DC
    DA_vec = np.ones(I + 1, dtype = np.float64) * setup.DA

    # calculate jacobian and invert it for the first points
    Jac = constJac(
                  I,
                  Dt,
                  Dx,
                  DC_vec,
                  DA_vec,
                  epsilon_vec,
                  setup.chi2,
                  M
                  )

    Jacinv = np.linalg.inv(Jac)
    
    # delete Jacobian - only inverse J is needed
    del Jac
    
    if setup.out_method == "full":
    
        # allocate output vectors
        
        sol = np.zeros([3 * I, N], dtype = np.float64)    

        sol[:,0] = sol_initial
        
        print("Start 'full' Simulation")
        
        for j in range(1,N):

            if j <= 20:
                print("Time Step: ", j)
            elif j > 20 and np.mod(j,100) == 0:
                print("Time Step: ", j)

            if j < 5:

                t_step_method = 1

                sol2 = sol[:,j-1]

            else:

                t_step_method = 2

                sol2 = sol[:,j-2]

            sol[:,j] = newton_krylov( lambda y: res(
                                        I,
                                        Dx,
                                        y,
                                        sol[:,j-1], # sol1 (j-1)
                                        sol2, # sol2 (j-2) for BDF2 only
                                        setup.chi1,
                                        setup.chi2,
                                        DC_vec,
                                        DA_vec,
                                        Dt,
                                        kA,
                                        kC,
                                        foxA,
                                        foxC,
                                        phiC[j],
                                        epsilon_vec,
                                        setup.model,
                                        t_step_method,
                                        M
                                        ),
                                        sol[:,j-1],
                                        inner_M = Jacinv,
                                        method = "lgmres",
                                        verbose = 0,
                                        maxiter = 100)

            # start current calculation if j >= 2, when the system reacts properly
            if j >= 2:

                # anodic current
                current[0,j] = calcAnodicCurrent( sol[2*I,j], sol[2*I,j-1], sol[2*I,j-2], Dt, Dx[0], setup.chi2 )

                # catodic current
                current[1,j] = calcCatodicCurrent( sol[3*I-1,j], sol[3*I-1,j-1], sol[3*I-1,j-2],
                                    phiC[j], phiC[j-1], phiC[j-2], Dt, Dx[I-1], setup.chi2 )
                    
        # save results
        print("Save Results")
        np.save(setup.testname + "_sol.npy", sol)
        np.save(setup.testname + "_current.npy", current)

        # current only output method
    if setup.out_method == "c_only":
        
        # allocate output vectors
        
        sol1 = np.zeros([3 * I], dtype = np.float64)
        sol2 = np.zeros([3 * I], dtype = np.float64)    
        sol3 = np.zeros([3 * I], dtype = np.float64)    

        sol2 = sol_initial

        print("Start 'c_only' Simulation")

        for j in range(1,N):

            if j <= 20:
                print("Time Step: ", j)
            elif j < 100 and j > 20 and np.mod(j,10) == 0:
                print("Time Step: ", j)
            elif j > 20 and np.mod(j,5000) == 0:
                print("Time Step: ", j)

            if j < 5:

                t_step_method = 1

            else:

                t_step_method = 2

            sol1 = newton_krylov( lambda y: res(
                                        I,
                                        Dx,
                                        y,
                                        sol2, # sol2 (j-1)
                                        sol3, # sol3 (j-2) for BDF2 only
                                        setup.chi1,
                                        setup.chi2,
                                        DC_vec,
                                        DA_vec,
                                        Dt,
                                        kA,
                                        kC,
                                        foxA,
                                        foxC,
                                        phiC[j],
                                        epsilon_vec,
                                        setup.model,
                                        t_step_method,
                                        M
                                        ),
                                        sol1,
                                        inner_M = Jacinv,
                                        method = "lgmres",
                                        verbose = 0,
                                        maxiter = 100)

            # start current calculation if j >= 2, when the system reacts properly
            if j >= 2:

                # anodic current
                current[0,j] = calcAnodicCurrent( sol1[2*I], sol2[2*I], sol3[2*I], Dt, Dx[0], setup.chi2 )

                # catodic current
                current[1,j] = calcCatodicCurrent( sol1[3*I-1], sol2[3*I-1], sol3[3*I-1],
                                    phiC[j], phiC[j-1], phiC[j-2], Dt, Dx[I-1], setup.chi2 )
            # step solution
            sol3  = sol2
            sol2 = sol1

        # save results
        print("Save Results")
        np.save(setup.testname + "_current.npy", current)
     
# end --------------------------------------------------------------------------------------------------------


# end function definitions---------------------------------------------------------------------------------

# Main
if __name__ == "__main__":

	# Setup
	stp1 = Setup( 	I = 1000,
		            N = 2 ** 20,
		            Dt = 1e-4,
		            T = 300,
		            L = 10,
		            lengthscale = 1e9,
		            c0_in = 1.0,
		            DA = 1.0,
		            DC = 1.0,
		            D0_in = 1e-12,
		            epsilon = 1.0,
		            epsilon_m = 80.0,
		            testname = "test1",
	             	model = 0  )

	# Setup
	stp2 = Setup( 	I = 1000,
	             	N = 2 ** 20,
		            Dt = 1e-4,
		            T = 300,
		            L = 20,
		            lengthscale = 1e9,
		            c0_in = 1.0,
		            DA = 1.0,
		            DC = 1.0,
	             	D0_in = 1e-12,
		            epsilon = 1.0,
	             	epsilon_m = 80.0,
	             	testname = "test2",
	             	model = 0,
	             	out_method = "c_only"  )

	# Setup
	stp2 = Setup( 	I = 1000,
	             	N = 2 ** 20,
	             	Dt = 1e-4,
	             	T = 300,
	             	L = 30,
	             	lengthscale = 1e9,
	             	c0_in = 1.0,
	             	DA = 1.0,
	             	DC = 1.0,
	             	D0_in = 1e-12,
	             	epsilon = 1.0,
	             	epsilon_m = 80.0,
		            testname = "test3",
	            	model = 0,
	             	out_method = "c_only"  )

	# Setup
	stp4 = Setup( 	I = 10,
	             	N = 2 ** 10,
	             	Dt = 1e-4,
	     	        T = 300,
	 	            L = 40,
	 	            lengthscale = 1e9,
		            c0_in = 1.0,
		            DA = 1.0,
		            DC = 1.0,
		            D0_in = 1e-12,
		            epsilon = 1.0,
		            epsilon_m = 80.0,
		            testname = "test4",
		            model = 0,
		            out_method = "c_only" )



	# reaction boundary condition
	reaction_bc = {
	            	"kA": 0.0,
	            	"kC": 0.0,
	            	"foxA": 0.0,
	            	"foxC": 0.0
	              }

	# load input voltage
	phiC = np.load( "input_volt_1.npy" )

	t1 = clock()

	# start multiprocessing
	p1 = mp.Process( target = solver, args = ( stp1, reaction_bc, phiC, ))
	p2 = mp.Process( target = solver, args = ( stp2, reaction_bc, phiC, ))
	p3 = mp.Process( target = solver, args = ( stp3, reaction_bc, phiC, ))
	p4 = mp.Process( target = solver, args = ( stp4, reaction_bc, phiC, ))


	# start subprocesses
	p1.start()
	p2.start()
	p3.start()
	p4.start()


	p1.join()
	p2.join()
	p3.join()
	p4.join()

	t2 = clock()
	print("Overall Simulation Time [s]: ", (t2 - t1)  )
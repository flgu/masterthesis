import numpy as np
cimport numpy as cnp
import time
from scipy.optimize import newton_krylov
from solver_tools import *



def solvePDE( I, x_, t_, sol_initial, phiC, DC, DA, epsilon, chi1, chi2, Dt, kA, kC, foxA, foxC):

	cdef:
		int j = 0

	t1 = time.clock()

	# init and preallocate
	sol = np.zeros([3*I,t_.size], dtype = np.float64)
	sol[:,0] = sol_initial

    # calculate jacobian and invert it for the first points
	Jac = calcJac(I, sol[:,0], x_, DC, DA, chi1, chi2, Dt )
	Jacinv = np.linalg.inv(Jac)

	for j in range(1,t_.size):

		if j <= 100:
			print("Time Step: ", j)
		elif j > 100 and np.mod(j,100) == 0:
			print("Time Step: ", j)

		sol[:,j] = newton_krylov( lambda y: residual_model1( I, x_, y, sol[:,j-1], chi1, chi2, DC, DA, Dt, kA,
           kC, foxA, foxC, phiC[j], epsilon), sol[:,j-1], inner_M = Jacinv, method = "lgmres")

#        if np.linalg.norm(sol[0:I,j] - sol[0:I,j-1],ord=2) < 1e-8:
#            print("Steady State after ",j," Time Steps")
#            break
#
#    # cut vectors
#    if j < t_.shape[0]:
#        t_ = np.delete(t_, np.s_[j:], axis = 0)
#        sol = np.delete(sol, np.s_[j:], axis = 1)
#        phiC = np.delete(phiC, np.s_[j:], axis = 0)

	t2 = time.clock()
	print("Simulation Runtime Old Implementation: ", t2-t1)

	return sol
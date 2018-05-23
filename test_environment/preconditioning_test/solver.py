import numpy as np
from scipy.optimize import newton_krylov
from solver_tools import residual, calcJac
import time

def PDEsolver( I, x_, t_, sol_initial, gI_, DC, DA, epsilon, chi1, chi2, Dt):

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

        # calculate Jacobian and invert it; freezed version, update every third step
        if np.mod(j,30) == 0:
            Jac = calcJac(I, sol[:,j], x_, DC, DA, chi1, chi2, Dt )
            Jacinv = np.linalg.inv(Jac)

        sol[:,j] = newton_krylov( lambda y: residual( I, x_, y, sol[:,j-1], chi1, chi2, DC, DA, Dt, 0.0, 0.0, 0.0, 0.0, gI_[j], epsilon),
                    sol[:,j-1], inner_M = Jacinv, method = "lgmres")

        if np.linalg.norm(sol[0:I,j] - sol[0:I,j-1],ord=2) < 1e-7:
            print("Steady State after ",j," Time Steps")
            break

    t2 = time.clock()
    print("Simulation Runtime Old Implementation: ", t2-t1)

    # cut solution vector if steady state is reached
    if j<=t_.size:
        sol = np.delete(sol, np.s_[j+1:],axis=1)
        t_  = np.delete(t_, np.s_[j+1:],axis=0)
        gI_ = np.delete(gI_, np.s_[j+1:],axis=0)

    return sol, t_, gI_

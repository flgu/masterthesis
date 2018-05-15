import numpy as np
from scipy.optimize import newton_krylov
import time
import imp
import os

from functions.solver_tools import *

def createAxis(I):

    x_ = np.zeros(I+1)
    Dx = 1.0/I
    for i in range(0,I+1):
        x_[i] = i*Dx

    # coordinates of cell centers
    x = np.zeros(I)
    for i in range(1,I+1):
        x[i-1] = x_[i-1]+(x_[i]-x_[i-1])/2

    return x_, x


def PDEsolver( I, x_, t_, sol_initial, phiC, DC, DA, epsilon, chi1, chi2, Dt, kA, kC, foxA, foxC):

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

        if np.linalg.norm(sol[0:I,j] - sol[0:I,j-1],ord=2) < 1e-8:
            print("Steady State after ",j," Time Steps")
            break

    # cut vectors
    if j < t_.shape[0]:
        t_ = np.delete(t_, np.s_[j:], axis = 0)
        sol = np.delete(sol, np.s_[j:], axis = 1)
        phiC = np.delete(phiC, np.s_[j:], axis = 0)

    t2 = time.clock()
    print("Simulation Runtime Old Implementation: ", t2-t1)

    return sol

def SweepSolver( I, x_, t_, sol_initial, phiC, DC, DA, epsilon, chi1, chi2, Dt, kA, kC, foxA, foxC):

    # preallocate current
    current = np.zeros([2,t_.shape[0]], dtype = np.float64)

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


        # calculate boundary fluxes and current density
        fA = foxA - kA * sol[0,j]
        fC = kC * solN[I-1,j] - foxC


        current[0,j] = fA - ( sol[2*I,j] - sol[2*I,j-1] ) /((x_[1]-x_[0])*Dt*chi2)
        current[1,j] = fC - ( phiC[j] - sol[3*I-1,j] - phiC[j-1] + sol[3*I-1,j-1] ) /((x_[I] - x_[I-1])*Dt*chi2)

        if np.linalg.norm(sol[0:I,j] - sol[0:I,j-1],ord=2) < 1e-8:
            print("Steady State after ",j," Time Steps")
            break

    # cut vectors
    if j < t_.shape[0]:
        sol = np.delete(sol, np.s_[j:], axis = 1)
        current = np.delete(current, np.s_[j:], axis = 1)

    t2 = time.clock()
    print("Simulation Runtime: ", t2-t1)

    return sol, current

def ImpedanceSolverModel0( I, x_, t_, sol_initial, phiC, DC, DA, epsilon, chi1, chi2, Dt):

    # preallocate current
    current = np.zeros([2,t_.shape[0]], dtype = np.float64)

    t1 = time.clock()

    # init and preallocate
    sol = np.zeros([3*I,t_.size], dtype = np.float64)
    sol[:,0] = sol_initial

    # calculate jacobian and invert it for the first points
    Jac = calcJac(I, np.zeros( sol[:,0].shape ), x_, DC, DA, chi1, chi2, Dt )
    Jacinv = np.linalg.inv(Jac)

    for j in range(1,t_.size):

        if j <= 100:
            print("Time Step: ", j)
        elif j > 100 and np.mod(j,100) == 0:
            print("Time Step: ", j)

        sol[:,j] = newton_krylov( lambda y: residual_model0( I, x_, y, sol[:,j-1], chi1, chi2, DC, DA, Dt,
        phiC[j], epsilon), sol[:,j-1], inner_M = Jacinv, method = "lgmres")

        current[0,j] = - ( sol[2*I,j] - sol[2*I,j-1] ) /((x_[1]-x_[0])*Dt*chi2)
        current[1,j] = - ( phiC[j] - sol[3*I-1,j] - phiC[j-1] + sol[3*I-1,j-1] ) /((x_[I] - x_[I-1])*Dt*chi2)

    t2 = time.clock()
    print("Simulation Runtime: ", t2-t1)

    return current


def ImpedanceSolver( I, x_, t_, sol_initial, phiC, DC, DA, epsilon, chi1, chi2, Dt, kA, kC, foxA, foxC):

    # preallocate current
    current = np.zeros([2,t_.shape[0]], dtype = np.float64)

    t1 = time.clock()

    # init and preallocate
    sol = np.zeros([3*I,t_.size], dtype = np.float64)
    sol[:,0] = sol_initial

    # calculate jacobian and invert it for the first points
    Jac = calcJac(I, np.zeros( sol[:,0].shape ), x_, DC, DA, chi1, chi2, Dt )
    Jacinv = np.linalg.inv(Jac)

    for j in range(1,t_.size):

        if j <= 100:
            print("Time Step: ", j)
        elif j > 100 and np.mod(j,100) == 0:
            print("Time Step: ", j)

        sol[:,j] = newton_krylov( lambda y: residual_model1( I, x_, y, sol[:,j-1], chi1, chi2, DC, DA, Dt, kA,
        kC, foxA, foxC, phiC[j], epsilon), sol[:,j-1], inner_M = Jacinv, method = "lgmres")


        # calculate boundary fluxes and current density
        fA = foxA - kA * sol[0,j]
        fC = kC * sol[I-1,j] - foxC


        current[0,j] = fA - ( sol[2*I,j] - sol[2*I,j-1] ) /((x_[1]-x_[0])*Dt*chi2)
        current[1,j] = fC - ( phiC[j] - sol[3*I-1,j] - phiC[j-1] + sol[3*I-1,j-1] ) /((x_[I] - x_[I-1])*Dt*chi2)

    t2 = time.clock()
    print("Simulation Runtime: ", t2-t1)

    return current
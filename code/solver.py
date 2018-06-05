import numpy as np
from scipy.optimize import newton_krylov
import time

from functions.residual import residual
import functions.system_tools as st
from functions.jacobian import constJac

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

def ImpedanceSolver_m0( I,
                       L,
                       c0,
                       epsilon_m,
                       phi0,
                       T,
                       N,
                       Dt,
                       M,
                       sol_initial,
                       phiC,
                       DC,
                       DA,
                       epsilon,
                       model,
                       kA,
                       foxA,
                       kC,
                       foxC ):

    # calculate numerical constants
    chi2 = st.calcChi1( phi0, T )
    chi1 = st.calcChi2( L, c0, phi0, epsilon_m )

    # create x axis for simulation
    xi = np.zeros(I+1, dtype = np.float64)
    x_ = np.zeros(I+1, dtype = np.float64)

    for i in range(0,I+1):

        xi[i] = i / I

    # creating x axis with functional relation
    x_ = xi
    #x_ = np.sin( ( np.pi * xi  ) / 2 ) ** 2

    # getting cell volumes
    Dx = np.zeros(I, dtype = np.float64)
    Dx = x_[1:] - x_[:I]

    # create time axis
    t_ = np.zeros(N)

    for j in range(0,N):

        t_[j] = j * Dt

    # preallocate current
    current = np.zeros([2, N], dtype = np.float64)
    epsilon_vec = np.ones(I + 1, dtype = np.float64) * epsilon
    DC_vec = np.ones(I + 1, dtype = np.float64) * DC
    DA_vec = np.ones(I + 1, dtype = np.float64) * DA

    t1 = time.clock()

    # init and preallocate
#    sol = np.zeros([3 * I,N], dtype = np.float64)

    sol1 = np.zeros([3 * I], dtype = np.float64)
    sol2 = np.zeros([3 * I], dtype = np.float64)


    sol1 = sol_initial

    # calculate jacobian and invert it for the first points
    Jac = constJac( I, M, Dt, Dx, DC_vec, DA_vec, epsilon_vec, chi2)
    Jacinv = np.linalg.inv(Jac)

    # delete Jacobian - only inverse J is needed
    #del Jac

    for j in range(1,N):

        if j <= 100:
            print("Time Step: ", j)
        elif j > 100 and np.mod(j,100) == 0:
            print("Time Step: ", j)

        # input --> sol1  --> output sol2

        sol2 = newton_krylov( lambda y: residual( I,
                                    Dx,
                                    y,
                                    sol1,
                                    chi1,
                                    chi2,
                                    DC_vec,
                                    DA_vec,
                                    Dt,
                                    M,
                                    phiC[j],
                                    epsilon_vec,
                                    model,
                                    kA,
                                    foxA,
                                    kC,
                                    foxC),
                                    sol1,
                                    inner_M = Jacinv,
                                    method = "lgmres",
                                    verbose = 0,
                                    maxiter = 100)

        current[0,j] = - ( sol2[2*I] - sol1[2*I] ) / (Dx[0] * Dt * chi2)
        current[1,j] = - ( phiC[j] - sol2[3*I-1] - phiC[j-1] + sol1[3*I-1] ) / (Dx[I-1] * Dt * chi2)

        # step solution sol1 = sol2 --> sol1 old solution j-1
        sol1 = sol2

    t2 = time.clock()
    print("Simulation Runtime: ", t2-t1)

    return current


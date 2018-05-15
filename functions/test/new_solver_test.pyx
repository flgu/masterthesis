import numpy as np
cimport numpy as cnp
cimport cython
from cython.view cimport array as cvarray

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def residual_model0_new( int I,
        cnp.ndarray[ cnp.float64_t, ndim=1] x_,
        cnp.ndarray[ cnp.float64_t, ndim=1] solN,
        cnp.ndarray[ cnp.float64_t, ndim=1] sol1,
        double chi1,
        double chi2,
        cnp.ndarray[ cnp.float64_t, ndim=1] DC,
        cnp.ndarray[ cnp.float64_t, ndim=1] DA,
        double Dt,
        double phiC,
        cnp.ndarray[ cnp.float64_t, ndim=1] epsilon ):
    '''
    Stand: 03.05.2018
    '''

    cdef:
        int i
        cnp.ndarray[ cnp.float64_t, ndim=1] residual = np.zeros(3*I, dtype = np.float64)

        # upwinding
        double velup = 0.0
        double veldown = 0.0
        double upfluxC = 0.0
        double upfluxA = 0.0
        double downfluxC = 0.0
        double downfluxA = 0.0

    '''
    BDF: c_j - c_j-1 + Dt/Dx_i (f_up - f_down)

    previous time step: c_j-1 = sol1

    Discretization of flux:
        ( fup - fdown ) / Dx_cell

    '''

    # Anode, upflux only, substitute downflux with boundary condition
    # calculate upwinding velocities
    velup = -chi1*2*(solN[2*I+1]-solN[2*I])/(x_[2]-x_[0])

    # calc upwinding
    # cations
    if DC[1]*velup >=0:
        upfluxC = solN[0]*velup* DC[1]
    else:
        upfluxC = solN[1]*velup* DC[1]

    # anions
    if -DA[1]*velup >=0:
        upfluxA = -solN[I]*velup*DA[1]
    else:
        upfluxA = -solN[I+1]*velup*DA[1]


    # cation
    residual[0] = ( solN[0] - sol1[0] + Dt/(x_[1]-x_[0])*( -DC[1]*2*(solN[1]-solN[0])/(x_[2]-x_[0]) + upfluxC ))

    # anion
    residual[I] = ( solN[I] - sol1[I] + Dt/(x_[1]-x_[0])*( -DA[1]*2*(solN[I+1]-solN[I])/(x_[2]-x_[0]) + upfluxA ) )

    # potential at x0 , boundary condition
    residual[2*I] = (( 2*epsilon[1]*(solN[2*I+1]-solN[2*I])/(x_[2]-x_[0]) - epsilon[0]*solN[2*I]/(x_[1]-x_[0]) )/(x_[1]-x_[0])
        + chi2*( solN[0] - solN[I]) )

    # inner points, loop over cell centers
    for i in range(1,I-1):

        # calculate upwinding velocities
        veldown = -chi1*2*(solN[2*I+i]-solN[2*I+i-1])/ (x_[i+1]-x_[i-1])
        velup = -chi1*2*(solN[2*I+i+1]-solN[2*I+i])/(x_[i+2]-x_[i])

        # calc upwinding
        # cations
        if DC[i]* veldown >=0:
            downfluxC = solN[i-1]*veldown* DC[i]
        else:
            downfluxC = solN[i]*veldown* DC[i]

        if DC[i+1]* velup >=0:
            upfluxC = solN[i]*velup* DC[i+1]
        else:
            upfluxC = solN[i+1]*velup* DC[i+1]

        # anions
        if -DA[i]*veldown >=0:
            downfluxA = -solN[I+i-1]*veldown*DA[i]
        else:
            downfluxA = -solN[I+i]*veldown*DA[i]

        if -DA[i+1]*velup >=0:
            upfluxA = -solN[I+i]*velup*DA[i+1]
        else:
            upfluxA = -solN[I+i+1]*velup*DA[i+1]

        # cations
        residual[i] = ( solN[i] - sol1[i] + Dt/( x_[i+1] - x_[i]  )*( -DC[i+1]*2*(solN[i+1]-solN[i])/(x_[i+2]-x_[i])
            +DC[i]*2*(solN[i]-solN[i-1])/(x_[i+1]-x_[i-1]) + upfluxC - downfluxC ))

        # anions shifted about I
        residual[I+i] = ( solN[I+i] - sol1[I+i] + Dt/( x_[i+1] - x_[i]  )*( -DA[i+1]*2*(solN[I+i+1]-solN[I+i])/(x_[i+2]-x_[i])
            + DA[i]*2*(solN[I+i]-solN[I+i-1])/(x_[i+1]-x_[i-1]) + upfluxA - downfluxA ))

        # potential equation from 1:I-2 --> two extra calculations needed
        residual[2*I+i] = (( 2*epsilon[i+1]*(solN[2*I+i+1] - solN[2*I+i])/(x_[i+2] - x_[i])
            -2*epsilon[i]*(solN[2*I+i] - solN[2*I+i-1])/(x_[i+1] - x_[i-1]) )/(x_[i+1]-x_[i]) + chi2*( solN[i] - solN[I+i]) )

    # catode, downflux only, substitute upflux with boundary condition
    # calc upwinding

    # calculate upwinding velocities
    veldown = -chi1*2*(solN[3*I-1]-solN[3*I-2])/(x_[I]-x_[I-2])

    # cations
    if DC[I-1]* veldown >=0:
        downfluxC = solN[I-2]*veldown* DC[I-1]
    else:
        downfluxC = solN[I-1]*veldown* DC[I-1]

    # anions
    if -DA[I-1]*veldown >=0:
        downfluxA = -solN[2*I-2]*veldown*DA[I-1]
    else:
        downfluxA = -solN[2*I-1]*veldown*DA[I-1]

    # catode boundary conditions
    # cations
    residual[I-1] = ( solN[I-1] - sol1[I-1] + Dt/(x_[I] - x_[I-1])*( + DC[I-1]*2*(solN[I-1]-solN[I-2])/(x_[I]-x_[I-2]) -downfluxC ))

    # anions
    residual[2*I-1] = ( solN[2*I-1] - sol1[2*I-1] + Dt/(x_[I] - x_[I-1])*( +DA[I-1]*2*(solN[2*I-1]-solN[2*I-2])/(x_[I]-x_[I-2]) - downfluxA ))

    # potential at right boundary
    residual[3*I-1] = (( epsilon[I-1]*(phiC - solN[3*I-1])/(x_[I] - x_[I-1])
        -epsilon[I]*2*(solN[3*I-1]-solN[3*I-2])/(x_[I]-x_[I-2]) )/(x_[I]-x_[I-1]) + chi2*( solN[I-1] - solN[2*I-1]))

    return residual
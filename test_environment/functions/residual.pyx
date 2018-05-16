import numpy as np
cimport numpy as cnp
cimport cython
from cython.view cimport array as cvarray

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

# ersetze x_ array durch memoryview
# ersetze solN durch memoryview
# ersetze sol1 durch memoryview

def residual_m0( int I,
        double [:] xview,
        double [:] solNview,
        double [:] sol1view,
        double chi1,
        double chi2,
        double [:] DCview,
        double [:] DAview,
        double Dt,
        double phiC,
        double [:] epsilonview ):
    '''
    Stand: 16.05.2018
    '''
    cdef:
        int i
        object[ double, ndim=1] residual = np.zeros(3*I, dtype = np.float64)
        # define memoryview on the residual
        double [:] res_view = residual

        # upwinding
        double velup = 0.0
        double veldown = 0.0
        double upfluxC = 0.0
        double upfluxA = 0.0
        double downfluxC = 0.0
        double downfluxA = 0.0

    '''
    BDF: c_j - c_j-1 + Dt/Dx_i (f_up - f_down)

    previous time step: c_j-1 = sol1view

    Discretization of flux:
        ( fup - fdown ) / Dx_cell

    '''

    # Anode, upflux only, substitute downflux with boundary condition
    # calculate upwinding velocities
    velup = -chi1*2*(solNview[2*I+1]-solNview[2*I])/(xview[2]-xview[0])

    # calc upwinding
    # cations
    if DCview[1]*velup >=0:
        upfluxC = solNview[0]*velup* DCview[1]
    else:
        upfluxC = solNview[1]*velup* DCview[1]

    # anions
    if -DAview[1]*velup >=0:
        upfluxA = -solNview[I]*velup*DAview[1]
    else:
        upfluxA = -solNview[I+1]*velup*DAview[1]


    # cation
    res_view[0] = ( solNview[0] - sol1view[0] + Dt/(xview[1]-xview[0])*( -DCview[1]*2*(solNview[1]-solNview[0])/(xview[2]-xview[0]) + upfluxC ))

    # anion
    res_view[I] = ( solNview[I] - sol1view[I] + Dt/(xview[1]-xview[0])*( -DAview[1]*2*(solNview[I+1]-solNview[I])/(xview[2]-xview[0]) + upfluxA ) )

    # potential at x0 , boundary condition
    res_view[2*I] = (( 2*epsilonview[1]*(solNview[2*I+1]-solNview[2*I])/(xview[2]-xview[0]) - epsilonview[0]*solNview[2*I]/(xview[1]-xview[0]) )/(xview[1]-xview[0])
        + chi2*( solNview[0] - solNview[I]) )

    # inner points, loop over cell centers
    for i in range(1,I-1):

        # calculate upwinding velocities
        veldown = -chi1*2*(solNview[2*I+i]-solNview[2*I+i-1])/ (xview[i+1]-xview[i-1])
        velup = -chi1*2*(solNview[2*I+i+1]-solNview[2*I+i])/(xview[i+2]-xview[i])

        # calc upwinding
        # cations
        if DCview[i]* veldown >=0:
            downfluxC = solNview[i-1]*veldown* DCview[i]
        else:
            downfluxC = solNview[i]*veldown* DCview[i]

        if DCview[i+1]* velup >=0:
            upfluxC = solNview[i]*velup* DCview[i+1]
        else:
            upfluxC = solNview[i+1]*velup* DCview[i+1]

        # anions
        if -DAview[i]*veldown >=0:
            downfluxA = -solNview[I+i-1]*veldown*DAview[i]
        else:
            downfluxA = -solNview[I+i]*veldown*DAview[i]

        if -DAview[i+1]*velup >=0:
            upfluxA = -solNview[I+i]*velup*DAview[i+1]
        else:
            upfluxA = -solNview[I+i+1]*velup*DAview[i+1]

        # cations
        res_view[i] = ( solNview[i] - sol1view[i] + Dt/( xview[i+1] - xview[i]  )*( -DCview[i+1]*2*(solNview[i+1]-solNview[i])/(xview[i+2]-xview[i])
            +DCview[i]*2*(solNview[i]-solNview[i-1])/(xview[i+1]-xview[i-1]) + upfluxC - downfluxC ))

        # anions shifted about I
        res_view[I+i] = ( solNview[I+i] - sol1view[I+i] + Dt/( xview[i+1] - xview[i]  )*( -DAview[i+1]*2*(solNview[I+i+1]-solNview[I+i])/(xview[i+2]-xview[i])
            + DAview[i]*2*(solNview[I+i]-solNview[I+i-1])/(xview[i+1]-xview[i-1]) + upfluxA - downfluxA ))

        # potential equation from 1:I-2 --> two extra calculations needed
        res_view[2*I+i] = (( 2*epsilonview[i+1]*(solNview[2*I+i+1] - solNview[2*I+i])/(xview[i+2] - xview[i])
            -2*epsilonview[i]*(solNview[2*I+i] - solNview[2*I+i-1])/(xview[i+1] - xview[i-1]) )/(xview[i+1]-xview[i]) + chi2*( solNview[i] - solNview[I+i]) )

    # catode, downflux only, substitute upflux with boundary condition
    # calc upwinding

    # calculate upwinding velocities
    veldown = -chi1*2*(solNview[3*I-1]-solNview[3*I-2])/(xview[I]-xview[I-2])

    # cations
    if DCview[I-1]* veldown >=0:
        downfluxC = solNview[I-2]*veldown* DCview[I-1]
    else:
        downfluxC = solNview[I-1]*veldown* DCview[I-1]

    # anions
    if -DAview[I-1]*veldown >=0:
        downfluxA = -solNview[2*I-2]*veldown*DAview[I-1]
    else:
        downfluxA = -solNview[2*I-1]*veldown*DAview[I-1]

    # catode boundary conditions
    # cations
    res_view[I-1] = ( solNview[I-1] - sol1view[I-1] + Dt/(xview[I] - xview[I-1])*( + DCview[I-1]*2*(solNview[I-1]-solNview[I-2])/(xview[I]-xview[I-2]) -downfluxC ))

    # anions
    res_view[2*I-1] = ( solNview[2*I-1] - sol1view[2*I-1] + Dt/(xview[I] - xview[I-1])*( +DAview[I-1]*2*(solNview[2*I-1]-solNview[2*I-2])/(xview[I]-xview[I-2]) - downfluxA ))

    # potential at right boundary
    res_view[3*I-1] = (( epsilonview[I-1]*(phiC - solNview[3*I-1])/(xview[I] - xview[I-1])
        -epsilonview[I]*2*(solNview[3*I-1]-solNview[3*I-2])/(xview[I]-xview[I-2]) )/(xview[I]-xview[I-1]) + chi2*( solNview[I-1] - solNview[2*I-1]))

    return residual
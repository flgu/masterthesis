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
        double [:] Dxview,
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
    velup = -chi1*2*(solNview[2*I+1] - solNview[2*I]) / (Dxview[1] + Dxview[0])

    # calc upwinding
    # cations
    if DCview[1] * velup >=0:
        
        upfluxC = velup * DCview[1] * solNview[0]
        
    else:
        
        upfluxC = velup * DCview[1] * solNview[1]

    # anions
    if -DAview[1]*velup >=0:
        
        upfluxA = -velup * DAview[1] * solNview[I]
        
    else:
        
        upfluxA = -velup * DAview[1] * solNview[I+1]
        
    # cation
    res_view[0] = ( solNview[0] - sol1view[0] 
    
    + Dt * ( -DCview[1] * 2.0 * (solNview[1]-solNview[0]) / (Dxview[0] + Dxview[1])  + upfluxC ) / Dxview[0] )

    # anion
    res_view[I] = ( solNview[I] - sol1view[I] 
    
    + Dt * ( -DAview[1] * 2 * (solNview[I+1]-solNview[I]) / (Dxview[1] + Dxview[0]) + upfluxA ) / Dxview[0] )

    # potential at x0 , boundary condition phiA = 0.0 oBdA
    res_view[2*I] = (( 2 * epsilonview[1]*(solNview[2*I+1]-solNview[2*I]) / (Dxview[0] + Dxview[1]) 
    
    - epsilonview[0]*solNview[2*I] / Dxview[0] ) / Dxview[0] + chi2*( solNview[0] - solNview[I]) )

    # inner points, loop over cell centers
    for i in range(1,I-1):
        
        #
        # Godunov scheme
        #
        
        # calculate upwinding velocities
        velup = -chi1 * 2.0 * (solNview[2*I+i+1] - solNview[2*I+i]) / (Dxview[i+1] + Dxview[i])
        veldown = -chi1 * 2.0 * (solNview[2*I+i] - solNview[2*I+i-1]) / (Dxview[i] + Dxview[i-1])

        # flux at rigth cell edge
        # for reconstruction: slope is 2 * (c_i+1 - c_i) / (Dx_i+1 + Dx_i)
        
        # cations
        if DCview[i+1] * velup >=0:
            
            upfluxC = velup * solNview[i] * DCview[i+1]
            
        else:
            
            upfluxC = velup * solNview[i+1] * DCview[i+1]
        
        # anions
        if -DAview[i+1] * velup >=0:
            
            upfluxA = -velup * solNview[I+i] * DAview[i+1]
            
        else:
            
            upfluxA = -velup * solNview[I+i+1] * DAview[i+1]
        
        # flux at left cell edge
        
        # cations
        if DCview[i] * veldown >=0:
            
            downfluxC = veldown * solNview[i-1] * DCview[i]
            
        else:
            
            downfluxC = veldown * solNview[i] * DCview[i]
            
        # anions
        if -DAview[i] * veldown >=0:
            
            downfluxA = -veldown * solNview[I+i-1] * DAview[i]
            
        else:
            
            downfluxA = -veldown * solNview[I+i] * DAview[i]

        # end Godunov / upwinding calculation ----------------------------------

        # cations
        res_view[i] = ( solNview[i] - sol1view[i] 
        
        + Dt * ( -DCview[i+1] * 2 * (solNview[i+1] - solNview[i]) / (Dxview[i+1] + Dxview[i])
        
        + DCview[i] * 2 * (solNview[i] - solNview[i-1]) / (Dxview[i] + Dxview[i-1]) + upfluxC - downfluxC ) / Dxview[i] )

        # anions shifted about I
        res_view[I+i] = ( solNview[I+i] - sol1view[I+i] 
        
        + Dt * ( -DAview[i+1] * 2 * (solNview[I+i+1] - solNview[I+i]) / (Dxview[i+1] + Dxview[i])
        
        + DAview[i] * 2 * (solNview[I+i] - solNview[I+i-1]) / (Dxview[i] + Dxview[i-1]) + upfluxA - downfluxA ) / Dxview[i])

        # potential equation from 1:I-2 --> two extra calculations needed
        res_view[2*I+i] = ( ( 2 * epsilonview[i+1] * (solNview[2*I+i+1] - solNview[2*I+i]) / (Dxview[i+1] + Dxview[i])
        
            -2 * epsilonview[i] * (solNview[2*I+i] - solNview[2*I+i-1]) / (Dxview[i] + Dxview[i-1]) ) 
        
        / Dxview[i] + chi2 * ( solNview[i] - solNview[I+i]) )

    # catode, downflux only, substitute upflux with boundary condition
    # calc upwinding

    # calculate upwinding velocities
    veldown = -chi1 * 2 * (solNview[3*I-1] - solNview[3*I-2]) / (Dxview[I-1] + Dxview[I-2])

    # cations
    if DCview[I-1] * veldown >=0:
        
        downfluxC = veldown * solNview[I-2] * DCview[I-1]
        
    else:
        
        downfluxC = veldown * solNview[I-1] * DCview[I-1]

    # anions
    if -DAview[I-1] * veldown >=0:
        
        downfluxA = -veldown * solNview[2*I-2] * DAview[I-1]
        
    else:
        
        downfluxA = -veldown * solNview[2*I-1] * DAview[I-1]

    # catode boundary conditions
    # cations
    res_view[I-1] = ( solNview[I-1] - sol1view[I-1]
    
    + Dt * ( DCview[I-1] * 2 * (solNview[I-1] - solNview[I-2]) / (Dxview[I-1] + Dxview[I-2]) -downfluxC ) / Dxview[I-1] )

    # anions
    res_view[2*I-1] = ( solNview[2*I-1] - sol1view[2*I-1]
    
    + Dt * ( DAview[I-1] * 2 * (solNview[2*I-1] - solNview[2*I-2]) / (Dxview[I-1] + Dxview[I-2]) - downfluxA ) / Dxview[I-1] )

    # potential at right boundary
    res_view[3*I-1] = ( ( epsilonview[I-1] * (phiC - solNview[3*I-1]) / Dxview[I-1]
    
        -epsilonview[I] * 2 * (solNview[3*I-1] - solNview[3*I-2]) / (Dxview[I-1] + Dxview[I-2]) ) / Dxview[I-1]
    
    + chi2*( solNview[I-1] - solNview[2*I-1]))

    return residual
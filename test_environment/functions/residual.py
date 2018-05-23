import numpy as np


def residual_m0( I,
        Dx,
        solN,
        sol1,
        chi1,
        chi2,
        DCview,
        double [:] DAview,
        double Dt,
        double M,
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
        
    
    Cell volume (one dimensional) of cell with index i is Dx[i].
    
    
    Nondimensionalization:
    ----------------------
    
    From nondimensionalozation of the time scale we have a parameter M multiplicated to
    time step Dt --> M * Dt. This is used to shorten the Impedance simulations.

    '''

    # Anode, upflux only, substitute downflux with boundary condition
    # calculate upwinding velocities
    velup = -chi1*2*(solN[2*I+1] - solN[2*I]) / (Dx[1] + Dx[0])

    # calc upwinding
    # cations
    if DCview[1] * velup >=0:
        
        upfluxC = velup * DCview[1] * solN[0]
        
    else:
        
        upfluxC = velup * DCview[1] * solN[1]

    # anions
    if -DAview[1]*velup >=0:
        
        upfluxA = -velup * DAview[1] * solN[I]
        
    else:
        
        upfluxA = -velup * DAview[1] * solN[I+1]
        
    # cation
    res_view[0] = ( solN[0] - sol1[0] 
    
    + M * Dt * ( -DCview[1] * 2.0 * (solN[1]-solN[0]) / (Dx[0] + Dx[1])  + upfluxC ) / Dx[0] )

    # anion
    res_view[I] = ( solN[I] - sol1[I] 
    
    + M * Dt * ( -DAview[1] * 2 * (solN[I+1]-solN[I]) / (Dx[1] + Dx[0]) + upfluxA ) / Dx[0] )

    # potential at x0 , boundary condition phiA = 0.0 oBdA
    res_view[2*I] = (( 2 * epsilonview[1]*(solN[2*I+1]-solN[2*I]) / (Dx[0] + Dx[1]) 
    
    - epsilonview[0]*solN[2*I] / Dx[0] ) / Dx[0] + chi2*( solN[0] - solN[I]) )

    # inner points, loop over cell centers
    for i in range(1,I-1):
        
        #
        # Godunov scheme
        #
        
        # calculate upwinding velocities
        velup = -chi1 * 2.0 * (solN[2*I+i+1] - solN[2*I+i]) / (Dx[i+1] + Dx[i])
        veldown = -chi1 * 2.0 * (solN[2*I+i] - solN[2*I+i-1]) / (Dx[i] + Dx[i-1])

        # flux at rigth cell edge
        # for reconstruction: slope is 2 * (c_i+1 - c_i) / (Dx_i+1 + Dx_i)
        
        # cations
        if DCview[i+1] * velup >=0:
            
            upfluxC = velup * solN[i] * DCview[i+1]
            
        else:
            
            upfluxC = velup * solN[i+1] * DCview[i+1]
        
        # anions
        if -DAview[i+1] * velup >=0:
            
            upfluxA = -velup * solN[I+i] * DAview[i+1]
            
        else:
            
            upfluxA = -velup * solN[I+i+1] * DAview[i+1]
        
        # flux at left cell edge
        
        # cations
        if DCview[i] * veldown >=0:
            
            downfluxC = veldown * solN[i-1] * DCview[i]
            
        else:
            
            downfluxC = veldown * solN[i] * DCview[i]
            
        # anions
        if -DAview[i] * veldown >=0:
            
            downfluxA = -veldown * solN[I+i-1] * DAview[i]
            
        else:
            
            downfluxA = -veldown * solN[I+i] * DAview[i]

        # end Godunov / upwinding calculation ----------------------------------

        # cations
        res_view[i] = ( solN[i] - sol1[i] 
        
        + M * Dt * ( -DCview[i+1] * 2 * (solN[i+1] - solN[i]) / (Dx[i+1] + Dx[i])
        
        + DCview[i] * 2 * (solN[i] - solN[i-1]) / (Dx[i] + Dx[i-1]) + upfluxC - downfluxC ) / Dx[i] )

        # anions shifted about I
        res_view[I+i] = ( solN[I+i] - sol1[I+i] 
        
        + M * Dt * ( -DAview[i+1] * 2 * (solN[I+i+1] - solN[I+i]) / (Dx[i+1] + Dx[i])
        
        + DAview[i] * 2 * (solN[I+i] - solN[I+i-1]) / (Dx[i] + Dx[i-1]) + upfluxA - downfluxA ) / Dx[i])

        # potential equation from 1:I-2 --> two extra calculations needed
        res_view[2*I+i] = ( ( 2 * epsilonview[i+1] * (solN[2*I+i+1] - solN[2*I+i]) / (Dx[i+1] + Dx[i])
        
            -2 * epsilonview[i] * (solN[2*I+i] - solN[2*I+i-1]) / (Dx[i] + Dx[i-1]) ) 
        
        / Dx[i] + chi2 * ( solN[i] - solN[I+i]) )

    # catode, downflux only, substitute upflux with boundary condition
    # calc upwinding

    # calculate upwinding velocities
    veldown = -chi1 * 2 * (solN[3*I-1] - solN[3*I-2]) / (Dx[I-1] + Dx[I-2])

    # cations
    if DCview[I-1] * veldown >=0:
        
        downfluxC = veldown * solN[I-2] * DCview[I-1]
        
    else:
        
        downfluxC = veldown * solN[I-1] * DCview[I-1]

    # anions
    if -DAview[I-1] * veldown >=0:
        
        downfluxA = -veldown * solN[2*I-2] * DAview[I-1]
        
    else:
        
        downfluxA = -veldown * solN[2*I-1] * DAview[I-1]

    # catode boundary conditions
    # cations
    res_view[I-1] = ( solN[I-1] - sol1[I-1]
    
    + M * Dt * ( DCview[I-1] * 2 * (solN[I-1] - solN[I-2]) / (Dx[I-1] + Dx[I-2]) -downfluxC ) / Dx[I-1] )

    # anions
    res_view[2*I-1] = ( solN[2*I-1] - sol1[2*I-1]
    
    + M * Dt * ( DAview[I-1] * 2 * (solN[2*I-1] - solN[2*I-2]) / (Dx[I-1] + Dx[I-2]) - downfluxA ) / Dx[I-1] )

    # potential at right boundary
    res_view[3*I-1] = ( ( epsilonview[I-1] * (phiC - solN[3*I-1]) / Dx[I-1]
    
        -epsilonview[I] * 2 * (solN[3*I-1] - solN[3*I-2]) / (Dx[I-1] + Dx[I-2]) ) / Dx[I-1]
    
    + chi2*( solN[I-1] - solN[2*I-1]))

    return residual
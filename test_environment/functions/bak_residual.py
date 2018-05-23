#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:47:39 2018

@author: desktop
"""
import numpy as np
from numba import jit, float64, int32
#                   I       Dx          solN       sol1         chi1    chi2     DC         DA
@jit( float64[:]( int32, float64[:], float64[:], float64[:], float64, float64, float64[:], float64[:],float64, float64, float64, float64[:] ))
def residual( I,
        Dx,
        solN,
        sol1,
        chi1,
        chi2,
        DC,
        DA,
        Dt,
        M,
        phiC,
        epsilon ):
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
    residual = np.empty(3*I, dtype = np.float64)

    # upwinding
    velup = 0.0
    veldown = 0.0
    upfluxC = 0.0
    upfluxA = 0.0
    downfluxC = 0.0
    downfluxA = 0.0

    # Anode, upflux only, substitute downflux with boundary condition
    # calculate upwinding velocities
    velup = -chi1 * 2 * (solN[2*I+1] - solN[2*I]) / (Dx[1] + Dx[0])

    # calc upwinding
    # cations
    if velup >=0:
        
        upfluxC = velup * DC[1] * solN[0]
        
    else:
        
        upfluxC = velup * DC[1] * solN[1]

    # anions
    if -velup >=0:
        
        upfluxA = -velup * DA[1] * solN[I]
        
    else:
        
        upfluxA = -velup * DA[1] * solN[I+1]
        
    # cation
    residual[0] = ( solN[0] - sol1[0] 
    
    + M * Dt * ( -DC[1] * 2.0 * (solN[1] - solN[0]) / (Dx[0] + Dx[1]) + upfluxC ) / Dx[0] )

    # anion
    residual[I] = ( solN[I] - sol1[I] 
    
    + M * Dt * ( -DA[1] * 2 * (solN[I+1] - solN[I]) / (Dx[1] + Dx[0]) + upfluxA ) / Dx[0] )

    # potential at x0 , boundary condition phiA = 0.0 oBdA
    residual[2*I] = (( 2 * epsilon[1] * (solN[2*I+1] - solN[2*I]) / (Dx[0] + Dx[1]) 
    
    - epsilon[0] * solN[2*I] / Dx[0] ) / Dx[0] + chi2 * ( solN[0] - solN[I]) )

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
        
        # switch between first order and second order with van leer limiter
        if (sol1[i] - sol1[i-1]) * (sol1[i+1] - sol1[i]) < 0:
            
            limiter = 0.0

        else:
            r = ( sol1[i] - sol1[i-1] ) / ( sol1[i+1] - sol1[i] )
            
            limiter = 2 * r / ( 1.0 + r )
            
        # cations
        if velup >=0:
            
            upfluxC = velup * DC[i+1] * ( solN[i]
                    + Dx[i] * limiter * ( solN[i+1] - solN[i] ) / (Dx[i+1] + Dx[i]) )
                
        else:
            
            upfluxC = velup * DC[i+1] * ( solN[i+1]
                    - Dx[i+1]  * limiter * ( solN[i+1] - solN[i] ) / (Dx[i+1] + Dx[i]) )
        
        # anions
        if -velup >=0:
            
            upfluxA = -velup * DA[i+1] * ( solN[I+i]
                    + Dx[i] * limiter * ( solN[I+i+1] - solN[I+i] ) / (Dx[i+1] + Dx[i]) )
            
        else:
            
            upfluxA = -velup * DA[i+1] * ( solN[I+i+1]
                        - Dx[i+1]  * limiter * ( solN[I+i+1] - solN[I+i] ) / (Dx[i+1] + Dx[i]) )
        
        # cations
        if veldown >=0:
            
            downfluxC = veldown * DC[i] ( solN[i-1] 
                        + Dx[i-1] * limiter * (solN[i] - solN[i-1]) / (Dx[i] + Dx[i-1]) )
            
        else:
            
            downfluxC = veldown * DC[i] * ( solN[i] 
                        - Dx[i] * limiter * ( solN[i] - solN[i-1]) / (Dx[i] + Dx[i-1]) )
                
        # anions
        if -veldown >=0:
            
            downfluxA = -veldown * DA[i] * ( solN[I+i-1] 
                        + Dx[i-1] * limiter * (solN[I+i] - solN[I+i-1]) / (Dx[i] + Dx[i-1]) )
                
        else:
            
            downfluxA = -veldown * DA[i] * ( solN[I+i] 
                            - Dx[i] * limiter * ( solN[I+i] - solN[I+i-1]) / (Dx[i] + Dx[i-1]) )
                
        # end Godunov / upwinding calculation ----------------------------------

        # cations
        residual[i] = ( solN[i] - sol1[i] 
        
        + M * Dt * ( -DC[i+1] * 2 * (solN[i+1] - solN[i]) / (Dx[i+1] + Dx[i])
        
        + DC[i] * 2 * (solN[i] - solN[i-1]) / (Dx[i] + Dx[i-1]) + upfluxC - downfluxC ) / Dx[i] )
        
        # anions shifted about I
        residual[I+i] = ( solN[I+i] - sol1[I+i] 
        
        + M * Dt * ( -DA[i+1] * 2 * (solN[I+i+1] - solN[I+i]) / (Dx[i+1] + Dx[i])
        
        + DA[i] * 2 * (solN[I+i] - solN[I+i-1]) / (Dx[i] + Dx[i-1]) + upfluxA - downfluxA ) / Dx[i])

        # potential equation from 1:I-2 --> two extra calculations needed
        residual[2*I+i] = ( ( 2 * epsilon[i+1] * (solN[2*I+i+1] - solN[2*I+i]) / (Dx[i+1] + Dx[i])
        
            -2 * epsilon[i] * (solN[2*I+i] - solN[2*I+i-1]) / (Dx[i] + Dx[i-1]) ) 
        
        / Dx[i] + chi2 * ( solN[i] - solN[I+i]) )

    # catode, downflux only, substitute upflux with boundary condition
    # calc upwinding

    # calculate upwinding velocities
    veldown = -chi1 * 2 * (solN[3*I-1] - solN[3*I-2]) / (Dx[I-1] + Dx[I-2])

    # cations
    if veldown >=0:
        
        downfluxC = veldown * DC[I-1] * solN[I-2]
        
    else:
        
        downfluxC = veldown * DC[I-1] * solN[I-1]

    # anions
    if -veldown >=0:
        
        downfluxA = -veldown * DA[I-1] * solN[2*I-2]
        
    else:
        
        downfluxA = -veldown * DA[I-1] * solN[2*I-1]

    # catode boundary conditions
    # cations
    residual[I-1] = ( solN[I-1] - sol1[I-1]
    
    + M * Dt * ( DC[I-1] * 2 * (solN[I-1] - solN[I-2]) / (Dx[I-1] + Dx[I-2]) -downfluxC ) / Dx[I-1] )

    # anions
    residual[2*I-1] = ( solN[2*I-1] - sol1[2*I-1]
    
    + M * Dt * ( DA[I-1] * 2 * (solN[2*I-1] - solN[2*I-2]) / (Dx[I-1] + Dx[I-2]) - downfluxA ) / Dx[I-1] )

    # potential at right boundary
    residual[3*I-1] = ( ( epsilon[I-1] * (phiC - solN[3*I-1]) / Dx[I-1]
    
        -epsilon[I] * 2 * (solN[3*I-1] - solN[3*I-2]) / (Dx[I-1] + Dx[I-2]) ) / Dx[I-1]
    
    + chi2*( solN[I-1] - solN[2*I-1]))

    return residual
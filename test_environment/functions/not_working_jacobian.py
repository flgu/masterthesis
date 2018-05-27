#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numba as nb

@nb.jit( nb.float64[:,:]( nb.int32, # I
                          #nb.float64, # M
                          nb.float64, # Dt
                          nb.float64[:], # Dx
                          nb.float64[:], # DC
                          nb.float64[:], # DA
                          nb.float64[:], # epsilon
                          nb.float64 # chi2
                          ) )
def constJac( I,
             #M,
              Dt,
              Dx,
              DC,
              DA,
              epsilon,
              chi2):
    """
        I,
            sol,
            x_,
            DC,
            DA,
            chi1,
            chi2,
            Dt,
            M):
    last update: 2018.05.24

    Inputs:


        chi2

    only constant parts of the Jacobian

    Nondimensionalization:
    ----------------------

    From nondimensionalozation of the time scale we have a parameter M multiplicated to
    time step Dt --> M * Dt. This is used to shorten the Impedance simulations.
    """

    Jac = np.zeros([3*I,3*I], dtype = np.float64)

    # A00
    Jac[0,0] = 1 +  Dt * 2 * ( DC[1] / (Dx[0] * (Dx[1] + Dx[0]) ) )
    Jac[0,1] = - Dt * 2 * DC[1] / (Dx[0] * (Dx[1] + Dx[0]) )

    # A11
    Jac[I,I] = 1 +  Dt * 2 * ( DA[1] / (Dx[0] * (Dx[1] + Dx[0]) ) )
    Jac[I,I+1] = -Dt * 2 * DA[1] / (Dx[0] * (Dx[1] + Dx[0]) )

    # A22
    Jac[2*I,2*I] = - ( 2 * epsilon[1] / (Dx[1] + Dx[0]) + epsilon[0] / Dx[0] ) / Dx[0]

    Jac[2*I,2*I+1] = 2 * epsilon[1] / ( Dx[0] * ( Dx[1] + Dx[0] ) )

    # A20 and A21
    Jac[2*I,0] = chi2
    Jac[2*I,I] = -chi2

    for i in range(1,I-1):

        # A00
        Jac[i,i-1] = -Dt * 2 * DC[i] / (Dx[i] * (Dx[i] + Dx[i-1]) )

        Jac[i,i] = 1 +   Dt * 2 * ( DC[i+1] / (Dx[i] * (Dx[i+1] + Dx[i]) )
                                + DC[i] / (Dx[i] * (Dx[i] + Dx[i-1]) ) )

        Jac[i,i+1] = -Dt * 2 * DC[i+1] / (Dx[i] * (Dx[i+1] + Dx[i]) )

        # A11
        Jac[I+i,I+i-1] = -Dt * 2 * DA[i] / (Dx[i] * (Dx[i] + Dx[i-1]) )

        Jac[I+i,I+i] = 1 +  Dt * 2 * ( DA[i+1] / (Dx[i] * (Dx[i+1] + Dx[i]) )
                                + DA[i] / (Dx[i] * (Dx[i] + Dx[i-1]) ) )

        Jac[I+i,I+i+1] = -Dt * 2 * DA[i+1] / (Dx[i] * (Dx[i+1] + Dx[i]) )

        # A22
        Jac[2*I+i,2*I+i-1] = 2 * epsilon[i] / ( Dx[i] * ( Dx[i] + Dx[i-1] ) )

        Jac[2*I+i,2*I+i] = -2 * ( epsilon[i] / ( Dx[i] + Dx[i-1] )
                                + epsilon[i+1] / ( Dx[i+1] + Dx[i] )  ) / Dx[i]

        Jac[2*I+i,2*I+i+1] = 2 * epsilon[i+1] / ( Dx[i] * ( Dx[i+1] + Dx[i] ) )

        # A20 and A21
        Jac[2*I+i,i] = chi2
        Jac[2*I+i,I+i] = -chi2


    # A00
    Jac[I-1,I-2] = -Dt * 2 * DC[I-1] / (Dx[I-1] * (Dx[I-1] + Dx[I-2]) )
    Jac[I-1,I-1] = 1 +  Dt * 2 * ( DC[I-1] / (Dx[I-1] * (Dx[I-1] + Dx[I-2]) ) )

    # A11
    Jac[2*I-1,2*I-2] = -Dt * 2 * DA[I-1] / (Dx[I-1] * (Dx[I-1] + Dx[I-2]) )
    Jac[2*I-1,2*I-1] = 1 +  Dt * 2 * ( DA[I-1] / (Dx[I-1] * (Dx[I-1] + Dx[I-2]) ) )

    # A22
    Jac[3*I-1,3*I-2] = 2 * epsilon[I-1] / ( Dx[I-1] * ( Dx[I-1] + Dx[I-2] ) )

    Jac[3*I-1,3*I-1] = - ( 2 * epsilon[I-1] / ( Dx[I-1] + Dx[I-2] ) + epsilon[I] / Dx[I-1] ) /Dx[I-1]

    # A20 and A21
    Jac[3*I-1,I-1] = chi2
    Jac[3*I-1,2*I-1] = -chi2

    return Jac

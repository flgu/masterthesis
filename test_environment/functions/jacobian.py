#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import max as fmax
from numpy import min as fmin

def calcJac( I,
            sol,
            x_,
            DC,
            DA,
            chi1,
            chi2,
            Dt,
            M):
    """
    last update: 2018.05.24

    Nondimensionalization:
    ----------------------

    From nondimensionalozation of the time scale we have a parameter M multiplicated to
    time step Dt --> M * Dt. This is used to shorten the Impedance simulations.
    """

    Jac = np.zeros([3*I,3*I], dtype = np.float64)

        # calculate derivatives of boundary condition
    dfAdc = 0.0
    dfAdp = 0.0
    dfCdc = 0.0
    dfCdp = 0.0

    # loop over potential rows and fill jacobian for potential
    for i in range(0,I):

        # fill cation diagonal
        Jac[2*I+i,i] = chi2

        # fill anion diagonal
        Jac[2*I+i,I+i] = -chi2

        # fill tridiagonal for inner points
        if i > 0 and i < I-1:

            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # fill lower diag
            Jac[2*I+i,2*I+i-1] = 1.0/(x_[i+1] - x_[i-1]) * 2.0/(x_[i+1] - x_[i])

            # fill diag
            Jac[2*I+i,2*I+i] = ( -1.0/(x_[i+2] - x_[i]) - 1.0/(x_[i+1] - x_[i-1]) ) * 2.0/(x_[i+1] - x_[i])

            # fill upper diag
            Jac[2*I+i,2*I+i+1] = 1.0/(x_[i+2] - x_[i]) * 2.0/(x_[i+1] - x_[i])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # fill lower diag
            Jac[i,i-1] = M * Dt * ( -2*DC[i]/(x_[i+1] - x_[i-1]) + DC[i]*chi1*2* fmax([sol[2*I+i] - sol[2*I+i-1],0.0])/(x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i])

            # fill diag
            Jac[i,i] = ( 1 + M * Dt * ( 2*DC[i+1]/(x_[i+2] - x_[i]) + 2*DC[i]/(x_[i+1] - x_[i-1])
                               - DC[i+1]*chi1*2* fmax([sol[2*I+i+1]- sol[2*I+i],0.0]) / (x_[i+2] - x_[i])
                               + DC[i]*chi1*2* fmin([sol[2*I+i] - sol[2*I+i-1],0.0]) / (x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i]) )

            # fill upper diag
            Jac[i,i+1] = M * Dt * ( -2*DC[i+1]/(x_[i+2] - x_[i]) - DC[i+1]*chi1*2 * fmin([sol[2*I+i+1] - sol[2*I+i], 0.0]) / (x_[i+2] - x_[i]) ) / (x_[i+1] - x_[i])

            #--- dNP- / dc--------------------------------------------------------------------------------
            # fill lower diag
            Jac[I+i,I+i-1] = M * Dt * ( -2*DA[i]/(x_[i+1] - x_[i-1]) - DA[i]*chi1*2* fmax([sol[2*I+i] - sol[2*I+i-1], 0.0])/(x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i])

            # fill diag
            Jac[I+i,I+i] = ( 1 + M * Dt * ( 2*DA[i+1]/(x_[i+2] - x_[i]) + 2*DA[i]/(x_[i+1] - x_[i-1])
                               + DA[i+1]*chi1*2* fmax([sol[2*I+i+1]- sol[2*I+i], 0.0]) / (x_[i+2] - x_[i])
                               - DA[i]*chi1*2* fmin([sol[2*I+i] - sol[2*I+i-1], 0.0]) / (x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i]) )

            # fill upper diag
            Jac[I+i,I+i+1] = M * Dt * ( -2*DA[i+1]/(x_[i+2] - x_[i]) + DA[i+1]*chi1*2 * fmin([sol[2*I+i+1] - sol[2*I+i], 0.0]) / (x_[i+2] - x_[i]) ) / (x_[i+1] - x_[i])

            #--- dNP+- / dp--------------------------------------------------------------------------------
            # fill lower diag
            if not (sol[2*I+i] - sol[2*I+i-1]) == 0:

                 #--- dNP+ / dp--------------------------------------------------------------------------------
                Jac[i,2*I+i-1] =( M * Dt*(-DC[i]*chi1*( sol[i-1]*fmax([sol[2*I+i] - sol[2*I+i-1], 0.0])
                            + sol[i]*fmin([sol[2*I+i] - sol[2*I+i-1], 0.0]))*2/((x_[i+1] - x_[i-1]) * (sol[2*I+i] - sol[2*I+i-1])) ) / (x_[i+1] - x_[i]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                Jac[I+i,2*I+i-1] = ( M * Dt*(DA[i]*chi1*( sol[I+i-1] * fmax([sol[2*I+i] - sol[2*I+i-1], 0.0])
                            + sol[I+i] * fmin([sol[2*I+i] - sol[2*I+i-1], 0.0]))*2/((x_[i+1] - x_[i-1]) * (sol[2*I+i] - sol[2*I+i-1])) ) / (x_[i+1] - x_[i]) )

            # fill diag
            #--- dNP- / dp--------------------------------------------------------------------------------
            if not (sol[2*I+i+1] - sol[2*I+i]) == 0 and (sol[2*I+i] - sol[2*I+i-1]) == 0:

                #--- dNP+ / dp--------------------------------------------------------------------------------
                Jac[i,2*I+i] = ( M * Dt*(DC[i+1]*chi1*( sol[i] * fmax([sol[2*I+i+1] - sol[2*I+i], 0.0])
                        + sol[i+1] * fmin([sol[2*I+i+1] - sol[2*I+i], 0.0]) )*2*( (x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i])) )/ (x_[i+1] - x_[i]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                Jac[I+i,2*I+i] = ( M * Dt*(-DA[i+1]*chi1*( sol[I+i] * fmax([sol[2*I+i+1] - sol[2*I+i], 0.0])
                        + sol[I+i+1] * fmin([sol[2*I+i+1] - sol[2*I+i], 0.0]) )*2*( (x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i])) )/ (x_[i+1] - x_[i]) )

            elif not (sol[2*I+i] - sol[2*I+i-1]) == 0 and (sol[2*I+i+1] - sol[2*I+i]) == 0:

                #--- dNP+ / dp--------------------------------------------------------------------------------
                Jac[i,2*I+i] = ( M * Dt*(DC[i]*chi1*( sol[i-1] * fmax([sol[2*I+i] - sol[2*I+i-1], 0.0])
                        + sol[i] * fmin([sol[2*I+i] - sol[2*I+i-1], 0.0]))*2*( (x_[i+1] - x_[i-1])*(sol[2*I+i] - sol[2*I+i-1])) )/ (x_[i+1] - x_[i]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                Jac[I+i,2*I+i] = ( M * Dt*(-DA[i]*chi1*( sol[I+i-1] * fmax([sol[2*I+i] - sol[2*I+i-1], 0.0])
                        + sol[I+i] * fmin([sol[2*I+i] - sol[2*I+i-1], 0.0]))*2*( (x_[i+1] - x_[i-1])*(sol[2*I+i] - sol[2*I+i-1])) )/ (x_[i+1] - x_[i]) )

            elif not (sol[2*I+i+1] - sol[2*I+i]) == 0 and not (sol[2*I+i] - sol[2*I+i-1]) == 0:

                #--- dNP+ / dp--------------------------------------------------------------------------------
                Jac[i,2*I+i] = ( M * Dt*(DC[i+1]*chi1*( sol[i] * fmax([sol[2*I+i+1] - sol[2*I+i], 0.0])
                        + sol[i+1] * fmin([sol[2*I+i+1] - sol[2*I+i], 0.0]) )*2*( (x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i]))
                        + DC[i]*chi1*( sol[i-1] * fmax([sol[2*I+i] - sol[2*I+i-1], 0.0])
                        + sol[i] * fmin([sol[2*I+i] - sol[2*I+i-1], 0.0]))*2*( (x_[i+1] - x_[i-1])*(sol[2*I+i] - sol[2*I+i-1])) )/ (x_[i+1] - x_[i]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                Jac[I+i,2*I+i] = ( M * Dt*(-DA[i+1]*chi1*( sol[I+i] * fmax([sol[2*I+i+1] - sol[2*I+i], 0.0])
                        + sol[I+i+1] * fmin([sol[2*I+i+1] - sol[2*I+i], 0.0]) )*2*( (x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i]))
                        - DA[i]*chi1*( sol[I+i-1] * fmax([sol[2*I+i] - sol[2*I+i-1], 0.0])
                        + sol[I+i] * fmin([sol[2*I+i] - sol[2*I+i-1], 0.0]))*2*( (x_[i+1] - x_[i-1])*(sol[2*I+i] - sol[2*I+i-1])) )/ (x_[i+1] - x_[i]) )

            # fill upper diag
            if not (sol[2*I+i+1] - sol[2*I+i]) == 0:

                #--- dNP+ / dp--------------------------------------------------------------------------------
                Jac[i,2*I+i+1] = ( M * Dt*(-DC[i+1]*chi1*(sol[i] * fmax([sol[2*I+i+1] - sol[2*I+i], 0.0])
                        + sol[i+1] * fmin([sol[2*I+i+1] - sol[2*I+i], 0.0]))*2/((x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i])) ) / (x_[i+1] - x_[i]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                Jac[I+i,2*I+i+1] = ( M * Dt*(DA[i+1]*chi1*(sol[I+i] * fmax([sol[2*I+i+1] - sol[2*I+i], 0.0])
                        + sol[I+i+1] * fmin([sol[2*I+i+1] - sol[2*I+i], 0.0]) )*2/((x_[i+2] - x_[i])*(sol[2*I+i+1] - sol[2*I+i])) ) / (x_[i+1] - x_[i]) )

        if i == 0:

            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # dphi0
            Jac[2*I,2*I] = (-2.0/(x_[2] - x_[0]) - 1.0/(x_[1] - x_[0])) * 1.0/(x_[1] - x_[0])

            # dphi1
            Jac[2*I,2*I+1] =  2.0/(x_[2] - x_[0]) * 1.0/(x_[1] - x_[0])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # dc0
            Jac[0,0] = (1 + M * Dt * ( 2*DC[1]/(x_[2] - x_[0]) -DC[1]*chi1*2* fmax([sol[2*I+1] - sol[2*I], 0.0])/(x_[2] - x_[0])
                                - dfAdc ) / (x_[1] - x_[0]) )

            # dc1
            Jac[0,1] = M * Dt * (-2*DC[1]/(x_[2] - x_[0]) -DC[1]*chi1*2* fmin([sol[2*I+1] - sol[2*I], 0.0])/(x_[2] - x_[0]) ) / (x_[1] - x_[0])

            #--- dNP- / dc--------------------------------------------------------------------------------
            # dc0
            Jac[I,I] = (1 + M * Dt * ( 2*DA[1]/(x_[2] - x_[0]) + DA[1]*chi1*2* fmax([sol[2*I+1] - sol[2*I], 0.0])/(x_[2] - x_[0]) ) / (x_[1] - x_[0]) )

            # dc1
            Jac[I,I+1] = M * Dt * (-2*DA[1]/(x_[2] - x_[0]) + DA[1]*chi1*2* fmin([sol[2*I+1] - sol[2*I], 0.0])/(x_[2] - x_[0]) ) / (x_[1] - x_[0])

            if not (sol[2*I+1] - sol[2*I]) == 0:
                #--- dNP+ / dp--------------------------------------------------------------------------------
                # dp0
                Jac[0,2*I] = (M * Dt*( DC[1]*chi1*( sol[0] * fmax([sol[2*I+1] - sol[2*I], 0.0])
                    + sol[1] * fmin([sol[2*I+1] - sol[2*I], 0.0]) )*2/((x_[2] - x_[0])*(sol[2*I+1] - sol[2*I]))- dfAdp)/(x_[1] - x_[0]) )

                # dp1
                Jac[0,2*I+1] = (M * Dt*(-DC[1]*chi1*(sol[0] * fmax([sol[2*I+1] - sol[2*I], 0.0])
                    + sol[1] * fmin([sol[2*I+1] - sol[2*I], 0.0]) )*2/((x_[2] - x_[0])*(sol[2*I+1] - sol[2*I])) )/(x_[1] - x_[0]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                # dp0
                Jac[I,2*I] = (M * Dt*(-DA[1]*chi1*(sol[I] * fmax([sol[2*I+1] - sol[2*I], 0.0])
                    + sol[I+1] * fmin([sol[2*I+1] - sol[2*I], 0.0]) )*2/((x_[2] - x_[0])*(sol[2*I+1] - sol[2*I])))/(x_[1] - x_[0]) )

                # dp1
                Jac[I,2*I+1] = (M * Dt*(DA[1]*chi1*(sol[I] * fmax([sol[2*I+1] - sol[2*I], 0.0])
                    + sol[I+1] * fmin([sol[2*I+1] - sol[2*I], 0.0]) )*2/((x_[2] - x_[0])*(sol[2*I+1] - sol[2*I])) )/(x_[1] - x_[0]) )

        if i == I-1:

            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # dphiI-2
            Jac[3*I-1,3*I-2] = 2.0/(x_[I] - x_[I-2]) * 1.0/(x_[I] - x_[I-1])

            # dphiI-1
            Jac[3*I-1,3*I-1] = (-1.0/(x_[I] - x_[I-1]) - 2.0/(x_[I] - x_[I-2])) * 1.0/(x_[I] - x_[I-1])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # dcI-2
            Jac[I-1,I-2] = M * Dt * ( -2*DC[I-1]/(x_[I] - x_[I-2]) + DC[I-1]*chi1*2* fmax([sol[3*I-1] - sol[3*I-2], 0.0])/(x_[I] - x_[I-2]) ) / (x_[I] - x_[I-1])

            # dcI-1
            Jac[I-1,I-1] = (1+ M * Dt*( dfCdc + 2*DC[I-1]/(x_[I] - x_[I-2])
                          + DC[I-1]*chi1*2* fmin([sol[3*I-1] - sol[3*I-2], 0.0])/(x_[I] - x_[I-2]) )/(x_[I] - x_[I-1]) )

            #--- dNP- / dc--------------------------------------------------------------------------------
            # dcI-2
            Jac[2*I-1,2*I-2] = M * Dt * ( -2*DA[I-1]/(x_[I] - x_[I-2]) - DA[I-1]*chi1*2* fmax([sol[3*I-1] - sol[3*I-2], 0.0])/(x_[I] - x_[I-2]) ) / (x_[I] - x_[I-1])

            # dcI-1
            Jac[2*I-1,2*I-1] = (1+ M * Dt*( 2*DA[I-1]/(x_[I] - x_[I-2])
                          - DA[I-1]*chi1*2* fmin([sol[3*I-1] - sol[3*I-2], 0.0])/(x_[I] - x_[I-2]) )/(x_[I] - x_[I-1]) )


            if not (sol[3*I-1] - sol[3*I-2]) == 0:
                #--- dNP+ / dp--------------------------------------------------------------------------------
                # dpI-2
                Jac[I-1,2*I+I-2] = (M * Dt*(-DC[I-1]*chi1*( sol[I-2]* fmax([sol[3*I-1] - sol[3*I-2], 0.0])
                    + sol[I-1] * fmin([sol[3*I-1] - sol[3*I-2], 0.0]) )*2/((x_[I] - x_[I-2])*(sol[3*I-1] - sol[3*I-2])))/(x_[I] - x_[I-1]) )

                # dpI-1
                Jac[I-1,2*I+I-1] = (M * Dt*(dfCdp + DC[I-1]*chi1*(sol[I-2] * fmax([sol[3*I-1] - sol[3*I-2], 0.0])
                    + sol[I-1] * fmin([sol[3*I-1] - sol[3*I-2], 0.0]))*2/((x_[I] - x_[I-2])*(sol[3*I-1] - sol[3*I-2])) )/(x_[I] - x_[I-1]) )

                #--- dNP- / dp--------------------------------------------------------------------------------
                # dpI-2
                Jac[2*I-1,2*I+I-2] = (M * Dt*(DA[I-1]*chi1*(sol[2*I-2] * fmax([sol[3*I-1] - sol[3*I-2], 0.0])
                    + sol[2*I-1] * fmin([sol[3*I-1] - sol[3*I-2], 0.0]) )*2/((x_[I] - x_[I-2])*(sol[3*I-1] - sol[3*I-2])))/(x_[I] - x_[I-1]) )

                # dpI-1
                Jac[2*I-1,2*I+I-1] = (M * Dt*(-DA[I-1]*chi1*(sol[2*I-2] * fmax([sol[3*I-1] - sol[3*I-2], 0.0])
                    + sol[2*I-1] * fmin([sol[3*I-1] - sol[3*I-2], 0.0]) )*2/((x_[I] - x_[I-2])*(sol[3*I-1] - sol[3*I-2])) )/(x_[I] - x_[I-1]) )
    return Jac
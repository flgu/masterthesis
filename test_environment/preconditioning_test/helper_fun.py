import numpy as np

def calcJac(I, sol, x_, DC, DA, chi1, chi2, Dt ):

    M = np.zeros([3*I,3*I], dtype = np.float64)

    # calculate derivatives of boundary condition
    dfAdc = 0.0
    dfAdp = 0.0
    dfCdc = 0.0
    dfCdp = 0.0

    # loop over potential rows and fill jacobian for potential
    for i in range(0,I):

        # fill cation diagonal
        M[2*I+i,i] = chi2

        # fill anion diagonal
        M[2*I+i,I+i] = -chi2

        # fill tridiagonal for inner points
        if i > 0 and i < I-1:

            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # fill lower diag
            M[2*I+i,2*I+i-1] = 1.0/(x_[i+1] - x_[i-1]) * 2.0/(x_[i+1] - x_[i])

            # fill diag
            M[2*I+i,2*I+i] = ( -1.0/(x_[i+2] - x_[i]) - 1.0/(x_[i+1] - x_[i-1]) ) * 2.0/(x_[i+1] - x_[i])

            # fill upper diag
            M[2*I+i,2*I+i+1] = 1.0/(x_[i+2] - x_[i]) * 2.0/(x_[i+1] - x_[i])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # fill lower diag
            M[i,i-1] = Dt * ( -2*DC[i]/(x_[i+1] - x_[i-1]) + DC[i]*chi1*2* np.max([sol[2*I+i] - sol[2*I+i-1],0])/(x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i])
        
            # fill diag
            M[i,i] = ( 1 + Dt * ( 2*DC[i+1]/(x_[i+2] - x_[i]) + 2*DC[i]/(x_[i+1] - x_[i-1]) 
                               - DC[i+1]*chi1*2* np.max([sol[2*I+i+1]- sol[2*I+i],0]) / (x_[i+2] - x_[i])
                               + DC[i]*chi1*2* np.min([sol[2*I+i] - sol[2*I+i-1],0]) / (x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i]) )

            # fill upper diag
            M[i,i+1] = Dt * ( -2*DC[i+1]/(x_[i+2] - x_[i]) - DC[i+1]*chi1*2 * np.min([sol[2*I+i+1] - sol[2*I+i],0]) / (x_[i+2] - x_[i]) ) / (x_[i+1] - x_[i])

            #--- dNP- / dc--------------------------------------------------------------------------------
            # fill lower diag
            M[I+i,I+i-1] = Dt * ( -2*DA[i]/(x_[i+1] - x_[i-1]) - DA[i]*chi1*2* np.max([sol[2*I+i] - sol[2*I+i-1],0])/(x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i])
        
            # fill diag
            M[I+i,I+i] = ( 1 + Dt * ( 2*DA[i+1]/(x_[i+2] - x_[i]) + 2*DA[i]/(x_[i+1] - x_[i-1]) 
                               + DA[i+1]*chi1*2* np.max([sol[2*I+i+1]- sol[2*I+i],0]) / (x_[i+2] - x_[i])
                               - DA[i]*chi1*2* np.min([sol[2*I+i] - sol[2*I+i-1],0]) / (x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i]) )

            # fill upper diag
            M[I+i,I+i+1] = Dt * ( -2*DA[i+1]/(x_[i+2] - x_[i]) + DA[i+1]*chi1*2 * np.min([sol[2*I+i+1] - sol[2*I+i],0]) / (x_[i+2] - x_[i]) ) / (x_[i+1] - x_[i])

        if i == 0:

            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # dphi0
            M[2*I,2*I] = (-2.0/(x_[2] - x_[0]) - 1.0/(x_[1] - x_[0])) * 1.0/(x_[1] - x_[0])

            # dphi1
            M[2*I,2*I+1] =  2.0/(x_[2] - x_[0]) * 1.0/(x_[1] - x_[0])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # dc0
            M[0,0] = (1 + Dt * ( 2*DC[1]/(x_[2] - x_[0]) -DC[1]*chi1*2* np.max([sol[2*I+1] - sol[2*I],0])/(x_[2] - x_[0]) 
                                - dfAdc ) / (x_[1] - x_[0]) )
        
            # dc1
            M[0,1] = Dt * (-2*DC[1]/(x_[2] - x_[0]) -DC[1]*chi1*2* np.min([sol[2*I+1] - sol[2*I],0])/(x_[2] - x_[0]) ) / (x_[1] - x_[0])

            #--- dNP- / dc--------------------------------------------------------------------------------
            # dc0
            M[I,I] = (1 + Dt * ( 2*DA[1]/(x_[2] - x_[0]) + DA[1]*chi1*2* np.max([sol[2*I+1] - sol[2*I],0])/(x_[2] - x_[0]) ) / (x_[1] - x_[0]) )
        
            # dc1
            M[I,I+1] = Dt * (-2*DA[1]/(x_[2] - x_[0]) + DA[1]*chi1*2* np.min([sol[2*I+1] - sol[2*I],0])/(x_[2] - x_[0]) ) / (x_[1] - x_[0])

        if i == I-1:
        
            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # dphiI-2
            M[3*I-1,3*I-2] = 2.0/(x_[I] - x_[I-2]) * 1.0/(x_[I] - x_[I-1])

            # dphiI-1
            M[3*I-1,3*I-1] = (-1.0/(x_[I] - x_[I-1]) - 2.0/(x_[I] - x_[I-2])) * 1.0/(x_[I] - x_[I-1])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # dcI-2
            M[I-1,I-2] = Dt * ( -2*DC[I-1]/(x_[I] - x_[I-2]) + DC[I-1]*chi1*2* np.max([sol[3*I-1] - sol[3*I-2],0])/(x_[I] - x_[I-2]) ) / (x_[I] - x_[I-1])

            # dcI-1
            M[I-1,I-1] = (1+ Dt*( dfCdc + 2*DC[I-1]/(x_[I] - x_[I-2]) 
                          + DC[I-1]*chi1*2* np.min([sol[3*I-1] - sol[3*I-2],0])/(x_[I] - x_[I-2]) )/(x_[I] - x_[I-1]) )

             #--- dNP- / dc--------------------------------------------------------------------------------
            # dcI-2
            M[2*I-1,2*I-2] = Dt * ( -2*DA[I-1]/(x_[I] - x_[I-2]) - DA[I-1]*chi1*2* np.max([sol[3*I-1] - sol[3*I-2],0])/(x_[I] - x_[I-2]) ) / (x_[I] - x_[I-1])

            # dcI-1
            M[2*I-1,2*I-1] = (1+ Dt*( 2*DA[I-1]/(x_[I] - x_[I-2]) 
                          - DA[I-1]*chi1*2* np.min([sol[3*I-1] - sol[3*I-2],0])/(x_[I] - x_[I-2]) )/(x_[I] - x_[I-1]) )

    return M

def calc_approx_Jac(I, x_, DC, DA, chi1, chi2, Dt ):

    M = np.zeros([3*I,3*I], dtype = np.float64)

    # loop over potential rows and fill jacobian for potential
    for i in range(0,I):

        # fill cation diagonal
        M[2*I+i,i] = chi2

        # fill anion diagonal
        M[2*I+i,I+i] = -chi2

        # fill tridiagonal for inner points
        if i > 0 and i < I-1:

            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # fill lower diag
            M[2*I+i,2*I+i-1] = 1.0/(x_[i+1] - x_[i-1]) * 2.0/(x_[i+1] - x_[i])

            # fill diag
            M[2*I+i,2*I+i] = ( -1.0/(x_[i+2] - x_[i]) - 1.0/(x_[i+1] - x_[i-1]) ) * 2.0/(x_[i+1] - x_[i])

            # fill upper diag
            M[2*I+i,2*I+i+1] = 1.0/(x_[i+2] - x_[i]) * 2.0/(x_[i+1] - x_[i])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # fill lower diag
            M[i,i-1] = Dt * ( -2*DC[i]/(x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i])
        
            # fill diag
            M[i,i] = ( 1 + Dt * ( 2*DC[i+1]/(x_[i+2] - x_[i]) + 2*DC[i]/(x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i]) )

            # fill upper diag
            M[i,i+1] = Dt * ( -2*DC[i+1]/(x_[i+2] - x_[i]) ) / (x_[i+1] - x_[i])

            #--- dNP- / dc--------------------------------------------------------------------------------
            # fill lower diag
            M[I+i,I+i-1] = Dt * ( -2*DA[i]/(x_[i+1] - x_[i-1]) ) / (x_[i+1] - x_[i])
        
            # fill diag
            M[I+i,I+i] = ( 1 + Dt * ( 2*DA[i+1]/(x_[i+2] - x_[i]) + 2*DA[i]/(x_[i+1] - x_[i-1])  ) / (x_[i+1] - x_[i]) )

            # fill upper diag
            M[I+i,I+i+1] = Dt * ( -2*DA[i+1]/(x_[i+2] - x_[i])  ) / (x_[i+1] - x_[i])

        if i == 0:

            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # dphi0
            M[2*I,2*I] = (-2.0/(x_[2] - x_[0]) - 1.0/(x_[1] - x_[0])) * 1.0/(x_[1] - x_[0])

            # dphi1
            M[2*I,2*I+1] =  2.0/(x_[2] - x_[0]) * 1.0/(x_[1] - x_[0])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # dc0
            M[0,0] = (1 + Dt * ( 2*DC[1]/(x_[2] - x_[0])  ) / (x_[1] - x_[0]) )
        
            # dc1
            M[0,1] = Dt * (-2*DC[1]/(x_[2] - x_[0])  ) / (x_[1] - x_[0])

            #--- dNP- / dc--------------------------------------------------------------------------------
            # dc0
            M[I,I] = (1 + Dt * ( 2*DA[1]/(x_[2] - x_[0])  ) / (x_[1] - x_[0]) )
        
            # dc1
            M[I,I+1] = Dt * (-2*DA[1]/(x_[2] - x_[0])  ) / (x_[1] - x_[0])

        if i == I-1:
        
            #--- dP / dc & dP / dp------------------------------------------------------------------------
            # dphiI-2
            M[3*I-1,3*I-2] = 2.0/(x_[I] - x_[I-2]) * 1.0/(x_[I] - x_[I-1])

            # dphiI-1
            M[3*I-1,3*I-1] = (-1.0/(x_[I] - x_[I-1]) - 2.0/(x_[I] - x_[I-2])) * 1.0/(x_[I] - x_[I-1])

            #--- dNP+ / dc--------------------------------------------------------------------------------
            # dcI-2
            M[I-1,I-2] = Dt * ( -2*DC[I-1]/(x_[I] - x_[I-2])  ) / (x_[I] - x_[I-1])

            # dcI-1
            M[I-1,I-1] = (1+ Dt*( 2*DC[I-1]/(x_[I] - x_[I-2]) )/(x_[I] - x_[I-1]) )

             #--- dNP- / dc--------------------------------------------------------------------------------
            # dcI-2
            M[2*I-1,2*I-2] = Dt * ( -2*DA[I-1]/(x_[I] - x_[I-2]) ) / (x_[I] - x_[I-1])

            # dcI-1
            M[2*I-1,2*I-1] = ( 1+ Dt*( 2*DA[I-1]/(x_[I] - x_[I-2]) )/(x_[I] - x_[I-1]) )

    return M
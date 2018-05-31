import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def residual( 
               int I,
               double [:] Dx not None,
               double [:] solN not None, 
               double [:] sol1 not None,
               double [:] sol2 not None,
               double chi1,
               double chi2,
               double [:] DC not None, 
               double [:] DA not None,
               double Dt, 
               double kA, 
               double kC, 
               double foxA, 
               double foxC,
               double phiC,
               double [:] epsilon not None,
               int model,
               int t_step_method,
               double M
            ):
    '''
    Stand: 24.05.2018

    Inputs:
        I = number of control volumes, int32
        Dx = cell volumes, 1D array float64
        solN = timestep j solution vector, 1D array float64
        sol1 = timestep j-1 solution vector, 1D array float64
        chi1 = numerical constant, scalar, float64
        chi2 = numerical constant, scalar, float64
        DC = diffusion constant cations, 1D array float64
        DA = diffusion constant anions, 1D array float64
        Dt = constant time step, scalar float64
        M = nondimensionalization parameter for reference time, scalar float64
        phiC = catode voltage, scalar float64
        epsilon = permitivitty vector of solvent, 1D array float64
        model = declaration of model, scalar int32
        kA = bondary condition anode, scalar double64
        foxA = bondary condition anode, scala, double64
        kC = bondary condition catode, scalar double64
        foxC = bondary condition catode, scalar double64

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

    cdef:
        residual = np.empty(3*I, dtype = np.float64)
        double [:] res = residual
        double fA = 0.0
        double fC = 0.0
        double velup = 0.0
        double veldown = 0.0
        double upfluxC = 0.0
        double upfluxA = 0.0
        double downfluxC = 0.0
        double downfluxA = 0.0
        double limiter = 0.0
        double r = 0.0
        int i



    # boundary fluxes
    if model == 0:

        fA = 0.0
        fC = 0.0

    elif model == 1:

        # Potential difference between electrode and nearest solution
        fA = foxA - kA * solN[0]
        fC = kC * solN[I-1] - foxC
        
    elif model == 2:

        fA = 0.0
        fC = 0.0

    # upwinding
    

    # Anode, upflux only, substitute downflux with boundary condition
    # calculate upwinding velocities
    velup = -chi1 * 2.0 * (solN[2*I+1] - solN[2*I]) / (Dx[1] + Dx[0])
    
    # calc upwinding
    # cations
    if velup >= 0.0:

        upfluxC = DC[1] * solN[0] * velup

    else:

        upfluxC =  DC[1] * solN[1] * velup

    # anions
    if -velup >=0.0 :

        upfluxA = - DA[1] * solN[I] * velup

    else:

        upfluxA = - DA[1] * solN[I+1] * velup
    
    # distinguish BDF1 and BDF2
    if t_step_method == 1:

        # cation
        res[0] = ( solN[0] - sol1[0] + M * Dt * ( -DC[1] * 2.0 * (solN[1] - solN[0]) / (Dx[1] + Dx[0]) + upfluxC -fA ) / Dx[0] )
        
        # anion 
        res[I] = ( solN[I] - sol1[I] + M * Dt * ( -DA[1] * 2.0 * (solN[I+1] - solN[I]) / (Dx[1] + Dx[0]) + upfluxA ) / Dx[0] )

    elif t_step_method == 2:

        # cation
        res[0] = ( solN[0] - 4.0 * sol1[0] / 3.0 + sol2[0] / 3.0

                    + 2.0 * M * Dt * ( -DC[1] * 2.0 * (solN[1] - solN[0]) / (Dx[1] + Dx[0]) + upfluxC -fA ) / (3 * Dx[0]) )
        
        # anion 
        res[I] = ( solN[I] - 4.0 * sol1[I] / 3.0 + sol2[I] / 3.0

                    + 2.0 * M * Dt * ( -DA[1] * 2.0 * (solN[I+1] - solN[I]) / (Dx[1] + Dx[0]) + upfluxA ) / (3.0 * Dx[0]) )
    
    # potential at x0 , boundary condition
    res[2*I] = ( ( 2.0 * epsilon[1] * (solN[2*I+1] - solN[2*I]) / (Dx[1] + Dx[0]) - epsilon[0] * solN[2*I] / Dx[0] ) / Dx[0]
        + chi2 * ( solN[0] - solN[I]) )
     
    # inner points, loop over cell centers
    for i in range(1,I-1):
        
        # calculate upwinding velocities
        veldown = -chi1 * 2.0 * (solN[2*I+i] - solN[2*I+i-1]) / (Dx[i] + Dx[i-1])
        velup = -chi1 * 2.0 * (solN[2*I+i+1] - solN[2*I+i]) / (Dx[i+1] + Dx[i])
               
        # calc upwinding
        # cations
        if (sol1[i] - sol1[i-1]) * (sol1[i+1] - sol1[i]) < 1e-15:

            limiter = 0.0

        else:
            r = ( sol1[i] - sol1[i-1] ) / ( sol1[i+1] - sol1[i] )

            # van Leer limiter

            limiter = 2.0 * r / ( 1.0 + r )

        if veldown >= 0.0 :

            downfluxC = DC[i] * veldown  * ( solN[i-1]
                + Dx[i-1] * limiter * (solN[i] - solN[i-1]) / (Dx[i] + Dx[i-1]) )   

        else:

            downfluxC = DC[i] * veldown * ( solN[i]
                - Dx[i] * limiter * (solN[i] - solN[i-1]) / (Dx[i] + Dx[i-1]) )
            
        if velup >= 0.0 :

            upfluxC = DC[i+1] * velup * ( solN[i]
                + Dx[i] * limiter * ( solN[i+1] - solN[i] ) / (Dx[i+1] + Dx[i]) )

        else:

            upfluxC = DC[i+1] * velup * ( solN[i+1]
                - Dx[i+1]  * limiter * ( solN[i+1] - solN[i] ) / (Dx[i+1] + Dx[i]) )

        # anions
        if (sol1[I+i] - sol1[I+i-1]) * (sol1[I+i+1] - sol1[I+i]) < 1e-15:

            limiter = 0.0

        else:
            r = ( sol1[I+i] - sol1[I+i-1] ) / ( sol1[I+i+1] - sol1[I+i] )

            # van Leer limiter

            limiter = 2.0 * r / ( 1.0 + r )

        if -veldown >= 0.0 :

            downfluxA = - DA[i] * veldown * ( solN[I+i-1]
                + Dx[i-1] * limiter * (solN[I+i] - solN[I+i-1]) / (Dx[i] + Dx[i-1]) )

        else:

            downfluxA = - DA[i] * veldown * ( solN[I+i]
                    - Dx[i] * limiter * (solN[I+i] - solN[I+i-1]) / (Dx[i] + Dx[i-1]) )

        if -velup >= 0.0 :

            upfluxA = - DA[i+1] * velup * ( solN[I+i]
                    + Dx[i] * limiter * ( solN[I+i+1] - solN[I+i] ) / (Dx[i+1] + Dx[i]) )

        else:

            upfluxA = - DA[i+1] * velup * ( solN[I+i+1]
                    - Dx[i+1]  * limiter * ( solN[I+i+1] - solN[I+i] ) / (Dx[i+1] + Dx[i]) )
        
        # distinguish BDF1 and BDF2
        if t_step_method == 1:  
            # cations
            res[i] = ( solN[i] - sol1[i] + M * Dt * ( -DC[i+1] * 2.0 * (solN[i+1] - solN[i]) / (Dx[i+1]+ Dx[i])
                + DC[i] * 2.0 * (solN[i] - solN[i-1]) / (Dx[i] + Dx[i-1]) + upfluxC - downfluxC ) / Dx[i] ) 
            
            # anions shifted about I
            res[I+i] = ( solN[I+i] - sol1[I+i] + M * Dt * ( -DA[i+1] * 2.0 * (solN[I+i+1] - solN[I+i]) / (Dx[i+1] + Dx[i])
                + DA[i] * 2.0 * (solN[I+i] - solN[I+i-1]) / (Dx[i] + Dx[i-1]) + upfluxA - downfluxA ) / Dx[i] )
            
            # potential equation from 1:I-2 --> two extra calculations needed
            res[2*I+i] = ( ( 2.0 * epsilon[i+1] * (solN[2*I+i+1] - solN[2*I+i]) / ( Dx[i] * (Dx[i+1] + Dx[i]) )
                - 2.0 * epsilon[i] * (solN[2*I+i] - solN[2*I+i-1]) / ( Dx[i] * (Dx[i] + Dx[i-1]) ) ) + chi2 * ( solN[i] - solN[I+i]) )

        elif t_step_method == 2:

            # cations
            res[i] = ( solN[i] - 4.0 * sol1[i]  / 3.0 + sol2[i] / 3.0
                + 2.0 * M * Dt * ( -DC[i+1] * 2 * (solN[i+1] - solN[i]) / (Dx[i+1]+ Dx[i])
                + DC[i] * 2.0 * (solN[i] - solN[i-1]) / (Dx[i] + Dx[i-1]) + upfluxC - downfluxC ) / (3.0 * Dx[i]) ) 
            
            # anions shifted about I
            res[I+i] = ( solN[I+i] - 4.0 * sol1[I+i]  / 3.0 + sol2[I+i] / 3.0
                + 2.0 * M * Dt * ( -DA[i+1] * 2.0 * (solN[I+i+1] - solN[I+i]) / (Dx[i+1] + Dx[i])
                + DA[i] * 2.0 * (solN[I+i] - solN[I+i-1]) / (Dx[i] + Dx[i-1]) + upfluxA - downfluxA ) / (3.0 * Dx[i]) )
            
            # potential equation from 1:I-2 --> two extra calculations needed
            res[2*I+i] = ( ( 2.0 * epsilon[i+1] * (solN[2*I+i+1] - solN[2*I+i]) / ( Dx[i] * (Dx[i+1] + Dx[i]) )
                - 2.0 * epsilon[i] * (solN[2*I+i] - solN[2*I+i-1]) / ( Dx[i] * (Dx[i] + Dx[i-1]) ) ) + chi2 * ( solN[i] - solN[I+i]) )
       
    # catode, downflux only, substitute upflux with boundary condition
    # calc upwinding
    
    # calculate upwinding velocities
    veldown = - chi1 * 2.0 * (solN[3*I-1] - solN[3*I-2]) / (Dx[I-1] + Dx[I-2])
    
    # cations
    if veldown >= 0.0 :

        downfluxC = DC[I-1] * solN[I-2] * veldown

    else:

        downfluxC = DC[I-1] * solN[I-1] * veldown

    # anions
    if -veldown >= 0.0 :

        downfluxA = - DA[I-1] * solN[2*I-2] * veldown

    else:
        
        downfluxA = - DA[I-1] * solN[2*I-1] * veldown

    # catode boundary conditions
    # cations
    if t_step_method == 1: 

        res[I-1] = ( solN[I-1] - sol1[I-1] + M * Dt * ( DC[I-1] * 2.0 * (solN[I-1] - solN[I-2]) / (Dx[I-1] + Dx[I-2])
                + fC - downfluxC ) / Dx[I-1] )
        
        # anions
        res[2*I-1] = ( solN[2*I-1] - sol1[2*I-1] + M * Dt * ( DA[I-1] * 2.0 * (solN[2*I-1] - solN[2*I-2]) / (Dx[I-1] + Dx[I-2]) - downfluxA ) / Dx[I-1] )
        
    elif t_step_method == 2:

        res[I-1] = ( solN[I-1] - 4.0 * sol1[I-1] / 3.0 + sol2[I-1] / 3.0
            + 2.0 * M * Dt * ( DC[I-1] * 2.0 * (solN[I-1] - solN[I-2]) / (Dx[I-1] + Dx[I-2]) + fC - downfluxC ) / (3.0 * Dx[I-1]) )
        
        # anions
        res[2*I-1] = ( solN[2*I-1] - 4.0 * sol1[2*I-1]  / 3.0 + sol2[2*I-1] / 3.0
            + 2.0 * M * Dt * ( DA[I-1] * 2.0 * (solN[2*I-1] - solN[2*I-2]) / (Dx[I-1] + Dx[I-2]) - downfluxA ) / (3.0 * Dx[I-1])  )
        
    # potential at right boundary
    res[3*I-1] = (( epsilon[I-1] * (phiC - solN[3*I-1]) / ( Dx[I-1] * Dx[I-1])
        -epsilon[I] * 2.0 * (solN[3*I-1] - solN[3*I-2]) / ( Dx[I-1] * (Dx[I-1] + Dx[I-2]) ) ) + chi2 * ( solN[I-1] - solN[2*I-1]) )
    
    return residual

def constJac( I,
              Dt,
              Dx,
              DC,
              DA,
              epsilon,
              chi2,
              M
              ):
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
    Jac[0,0] = 1 + M * Dt * 2 * ( DC[1] / (Dx[0] * (Dx[1] + Dx[0]) ) )
    Jac[0,1] = - M * Dt * 2 * DC[1] / (Dx[0] * (Dx[1] + Dx[0]) )

    # A11
    Jac[I,I] = 1 +  M * Dt * 2 * ( DA[1] / (Dx[0] * (Dx[1] + Dx[0]) ) )
    Jac[I,I+1] = -M * Dt * 2 * DA[1] / (Dx[0] * (Dx[1] + Dx[0]) )

    # A22
    Jac[2*I,2*I] = - ( 2 * epsilon[1] / (Dx[1] + Dx[0]) + epsilon[0] / Dx[0] ) / Dx[0]

    Jac[2*I,2*I+1] = 2 * epsilon[1] / ( Dx[0] * ( Dx[1] + Dx[0] ) )

    # A20 and A21
    Jac[2*I,0] = chi2
    Jac[2*I,I] = -chi2

    for i in range(1,I-1):

        # A00
        Jac[i,i-1] = -M * Dt * 2 * DC[i] / (Dx[i] * (Dx[i] + Dx[i-1]) )

        Jac[i,i] = 1 +   M * Dt * 2 * ( DC[i+1] / (Dx[i] * (Dx[i+1] + Dx[i]) )
                                + DC[i] / (Dx[i] * (Dx[i] + Dx[i-1]) ) )

        Jac[i,i+1] = -M * Dt * 2 * DC[i+1] / (Dx[i] * (Dx[i+1] + Dx[i]) )

        # A11
        Jac[I+i,I+i-1] = -M * Dt * 2 * DA[i] / (Dx[i] * (Dx[i] + Dx[i-1]) )

        Jac[I+i,I+i] = 1 +  M * Dt * 2 * ( DA[i+1] / (Dx[i] * (Dx[i+1] + Dx[i]) )
                                + DA[i] / (Dx[i] * (Dx[i] + Dx[i-1]) ) )

        Jac[I+i,I+i+1] = -M * Dt * 2 * DA[i+1] / (Dx[i] * (Dx[i+1] + Dx[i]) )

        # A22
        Jac[2*I+i,2*I+i-1] = 2 * epsilon[i] / ( Dx[i] * ( Dx[i] + Dx[i-1] ) )

        Jac[2*I+i,2*I+i] = -2 * ( epsilon[i] / ( Dx[i] + Dx[i-1] )
                                + epsilon[i+1] / ( Dx[i+1] + Dx[i] )  ) / Dx[i]

        Jac[2*I+i,2*I+i+1] = 2 * epsilon[i+1] / ( Dx[i] * ( Dx[i+1] + Dx[i] ) )

        # A20 and A21
        Jac[2*I+i,i] = chi2
        Jac[2*I+i,I+i] = -chi2


    # A00
    Jac[I-1,I-2] = -M * Dt * 2 * DC[I-1] / (Dx[I-1] * (Dx[I-1] + Dx[I-2]) )
    Jac[I-1,I-1] = 1 + M * Dt * 2 * ( DC[I-1] / (Dx[I-1] * (Dx[I-1] + Dx[I-2]) ) )

    # A11
    Jac[2*I-1,2*I-2] = -M * Dt * 2 * DA[I-1] / (Dx[I-1] * (Dx[I-1] + Dx[I-2]) )
    Jac[2*I-1,2*I-1] = 1 + M * Dt * 2 * ( DA[I-1] / (Dx[I-1] * (Dx[I-1] + Dx[I-2]) ) )

    # A22
    Jac[3*I-1,3*I-2] = 2 * epsilon[I-1] / ( Dx[I-1] * ( Dx[I-1] + Dx[I-2] ) )

    Jac[3*I-1,3*I-1] = - ( 2 * epsilon[I-1] / ( Dx[I-1] + Dx[I-2] ) + epsilon[I] / Dx[I-1] ) /Dx[I-1]

    # A20 and A21
    Jac[3*I-1,I-1] = chi2
    Jac[3*I-1,2*I-1] = -chi2

    return Jac


def calcAnodicCurrent( double phiN, double phi1, double phi2, double Dt, double Dx, double chi2 ):

    cdef double current_A = 0.0
    # BDF 2
    current_A = -(3.0 * (phiN) - 4.0 * (phi1) + (phi2)) / (2.0 * Dt * Dx * chi2)

    return current_A 

def calcCatodicCurrent( double phiN, double phi1, double phi2, double phiCN, double phiC1,
                        double phiC2, double Dt, double Dx, double chi2 ):

    cdef double current_C = 0.0

    # BDF 2
    current_C = -(3.0 * (phiN - phiCN) - 4.0 * (phi1 - phiC1) + (phi2 - phiC2)) / (2.0 * Dt * Dx * chi2)

    return current_C
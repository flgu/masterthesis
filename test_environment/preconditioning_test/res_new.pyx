import numpy as np
cimport numpy as cnp
from libc.math cimport fmax, fmin

# Define constants
_EPSILON_VAC = 8.854187817e-12 # original value in [Farad/m],
_ELE_CHARGE = 1.6021766208e-19 # [Coulomb], constant value
_BOLTZMANN_CONST = 1.38064852e-23 # [Joule/Kelvin], constant value
_AVOGADRO_CONST = 6.022140857e23 # [1/mol]
_GAS_CONST = 8.314459848 # [J/(K*mol)]
_FARADAY_CONST = 96485.3328959 # [C/mol]

# calculation of numerical constants
def calcChi1( double phi0, double T):
    """
    Calculation of numerical constant chi1.

    Input:
        phi0 = reference potential in [V]
        T = temperature of the system in [K]
    """
    cdef:
        float _ELE_CHARGE = 1.6021766208e-19
        float _BOLTZMANN_CONST = 1.38064852e-23

    return _ELE_CHARGE*phi0 / (_BOLTZMANN_CONST * T)

def calcChi2( double L, double c0, double phi0, double epsilon_m):
    """
    Calculation of numerical constant chi2.

    Input:
        L = reference length of the system in [m]
        c0 = reference concentration of the system, in same length units as L in [1/m^3]
        phi0 = reference potential of the system in [V]
        epsilon_m = dielectric constant of the medium, dimensionless
    """
    cdef:
        double _EPSILON_VAC = 8.854187817e-12 # original value in [Farad/m],
        double _ELE_CHARGE = 1.6021766208e-19 # [Coulomb], constant value

    return L**2 * _ELE_CHARGE * c0 / (phi0*epsilon_m*_EPSILON_VAC)

def calcBjL( double epsilon_m, double T):
    """
    Calculate Bjerrum length.

    Input:
        T = temperature if the system, in Kelvin
        epsilon_m = dielectric constant of the medium, dimensionless
    """
    return _ELE_CHARGE**2 / (4*np.pi*_EPSILON_VAC*epsilon_m*_BOLTZMANN_CONST*T)

def calcDbL( double epsilon_m, double T, double I):
    """
    Calculate Debye length.

    Input:
        epsilon_m = dielectric constant of the medium, dimensionless.
        T = temperature of the system [Kelvin]
        I = ionic strength of the electrolyte in [1/m^3]
    """
    cdef:
        double _EPSILON_VAC = 8.854187817e-12 # original value in [Farad/m]
        double _BOLTZMANN_CONST = 1.38064852e-23 # [Joule/Kelvin], constant value
        double _ELE_CHARGE = 1.6021766208e-19 # [Coulomb], constant value

    return np.sqrt( (_EPSILON_VAC*epsilon_m*_BOLTZMANN_CONST*T)/(_ELE_CHARGE**2 * I))


def res_new( int I, cnp.ndarray[ cnp.float64_t, ndim=1] x_,
            cnp.ndarray[ cnp.float64_t, ndim=1] solN,
            cnp.ndarray[ cnp.float64_t, ndim=1] sol1,
            double chi1, double chi2,
            cnp.ndarray[ cnp.float64_t, ndim=1] DC,
            cnp.ndarray[ cnp.float64_t, ndim=1] DA,
            double Dt, double kC0, double kCI, double jC0, double jCI, double g0, double gI,
            cnp.ndarray[ cnp.float64_t, ndim=1] epsilon,
            str bctype):
    '''
    Stand: 03.04.2018, Potentialberechnung gefixt.

    Potential: just Dirichlet boundary conditions with anode value g0 and catode value gI
    vector with computational axis: x_
    Dx_i = (x_i+1 - x_i) für loop over cell center axis x

    Erfahrungen: Berechnung von DxV außerhalb bringt keine Performance Verbesserung
    '''

    cdef:
        int i
        cnp.ndarray[ cnp.float64_t, ndim=1] residual = np.zeros(3*I, dtype = np.float64)
        #cnp.ndarray[ cnp.float64_t, ndim=1] DxV = x_[1:]-x_[:-1]

        # boundary fluxes
        double fC0 = 0.0
        double fCI = 0.0

    '''
    BDF: c_j - c_j-1 + Dt/Dx_i (f_up - f_down)

    previous time step: c_j-1 = sol1

    Discretization of flux:
        ( fup - fdown ) / Dx_cell

    '''
    #

    # calculation of boundary fluxes
    if bctype == "BV":
        fC0 = jC0*np.exp((g0-solN[2*I])/2) - kC0*solN[0]*np.exp(-(g0-solN[2*I])/2)
        fCI = -(jCI*np.exp((gI-solN[3*I-1])/2) - kCI*solN[I-1]*np.exp(-(gI-solN[3*I-1])/2))

    elif bctype == "linear":
        fC0 =  jC0 - kC0*solN[0]
        fCI =  kCI*solN[I-1]  - jCI
    elif bctype == "blocking":
        fC0 = 0.0
        fCI = 0.0

    # Anode, upflux only, substitute downflux with boundary condition

    # cation (z = 1)
    residual[0] = ( solN[0] - sol1[0] + Dt/(x_[1]-x_[0])*( -DC[1] * 2 *( solN[1] - solN[0] ) / (x_[2] - x_[0])

            # upwinding upflux - boundary condition downflux
            -DC[1] * chi1 * ( fmax(2*(solN[2*I+1] - solN[2*I]) / (x_[2] - x_[0]),0.0) * solN[0]
            + fmin(2*(solN[2*I+1] - solN[2*I]) / (x_[2] - x_[0]),0.0) * solN[1] )  - fC0  ) )

#     # anion
    residual[I] = ( solN[I] - sol1[I] + Dt/(x_[1]-x_[0])*( -DA[1] * 2 * ( solN[I+1] - solN[I] ) / (x_[2] - x_[0])

            # upwinding upflux - boundary condition downflux
            +DA[1] * chi1 * ( fmax(2*(solN[2*I+1] - solN[2*I]) / (x_[2] - x_[0]),0.0) * solN[I]
            + fmin(2*(solN[2*I+1] - solN[2*I]) / (x_[2] - x_[0]),0.0) * solN[I+1]) ) )

    # potential at x0 , boundary condition
    residual[2*I] = ((
        # up
        2*epsilon[1]*(solN[2*I+1]-solN[2*I])/(x_[2]-x_[0])

                    # new boundary condition just Dirichlet, - down
           -epsilon[0]*(solN[2*I] - g0)/(x_[1]-x_[0])

            )/(x_[1]-x_[0])

        # down boundary condition
        + chi2*(solN[0] - solN[I])
        )

    # inner points, loop over cell centers
    for i in range(1,I-1):

        # cations
        residual[i] = ( solN[i] - sol1[i] + Dt/( x_[i+1] - x_[i]  )*(

            # diffusion
            -DC[i+1]  * 2 * (solN[i+1] - solN[i]) / (x_[i+2]-x_[i]) + DC[i] * 2 * (solN[i] - solN[i-1]) / (x_[i+1]-x_[i-1])

            # upwinding
            -DC[i+1] * chi1 * ( fmax(2*( solN[2*I+i+1] - solN[2*I+i]) / (x_[i+2] - x_[i]),0.0) * solN[i]
            + fmin(2*( solN[2*I+i+1] - solN[2*I+i]) / (x_[i+2] - x_[i]),0.0) * solN[i+1] )

            +DC[i] * chi1 * ( fmax(2*(solN[2*I+i] - solN[2*I+i-1])/ (x_[i+1] - x_[i-1]),0.0) * solN[i-1]
            + fmin(2*(solN[2*I+i] - solN[2*I+i-1])/ (x_[i+1] - x_[i-1]),0.0) * solN[i] )  ) )

        # anions shifted about I
        residual[I+i] = ( solN[I+i] - sol1[I+i] + Dt/( x_[i+1] - x_[i]  )
            *(
            # diffusion
            -DA[i+1]*2*(solN[I+i+1]-solN[I+i])/(x_[i+2]-x_[i]) #(DxV[i+1]+DxV[i]) # # diffusive upflux
            +DA[i]*2*(solN[I+i]-solN[I+i-1])/(x_[i+1]-x_[i-1]) #(DxV[i]+DxV[i-1])  # # diffusive downflux

            # upwinding(sign zA = -1 !)
            +DA[i+1] * chi1 * ( fmax(2*( solN[2*I+i+1] - solN[2*I+i]) / (x_[i+2] - x_[i]),0.0) * solN[I+i]
            + fmin(2*( solN[2*I+i+1] - solN[2*I+i]) / (x_[i+2] - x_[i]),0.0) * solN[I+i+1] )

            -DA[i] * chi1 *(fmax(2*(solN[2*I+i] - solN[2*I+i-1])/ (x_[i+1] - x_[i-1]),0.0) * solN[I+i-1]
            + fmin(2*(solN[2*I+i] - solN[2*I+i-1])/ (x_[i+1] - x_[i-1]),0.0) * solN[I+i] )  ) )

        # potential shifted about 2*I
        residual[2*I+i] = ((
            # upflux
            2*epsilon[i+1]*(solN[2*I+i+1] - solN[2*I+i])/(x_[i+2] - x_[i])#(DxV[i+1]+DxV[i])  #
            # downflux
            -2*epsilon[i]*(solN[2*I+i] - solN[2*I+i-1])/(x_[i+1] - x_[i-1]) )/(x_[i+1]-x_[i]) # (DxV[i] + DxV[i-1]) )/(DxV[i])    #
            +chi2*(solN[i] - solN[I+i])
            )

    # catode boundary conditions

    # cations
    residual[I-1] = ( solN[I-1] - sol1[I-1] + Dt/(x_[I] - x_[I-1])*(

            # diffusion downflux
            +DC[I-1] * 2 * (solN[I-1] - solN[I-2]) / (x_[I] - x_[I-2])

            # upwinding
            +fCI + DC[I-1] * chi1 * ( fmax(2*(solN[3*I-1] - solN[3*I-2]) / (x_[I] - x_[I-2]),0.0) * solN[I-2]
            + fmin(2*(solN[3*I-1] - solN[3*I-2]) / (x_[I] - x_[I-2]),0.0) * solN[I-1]) ) )

    # anions
    residual[2*I-1] = ( solN[2*I-1] - sol1[2*I-1]
            #+ Dt/(DxV[I-1])*( #
            + Dt/(x_[I] - x_[I-1])*(
            # diffusion downflux
            +DA[I-1]*2*(solN[2*I-1]-solN[2*I-2])/(x_[I]-x_[I-2]) # (DxV[I-1]+DxV[I-2]) #

            # upwinding
            - DA[I-1] * chi1 * ( fmax(2*(solN[3*I-1] - solN[3*I-2])/ (x_[I] - x_[I-2]),0.0) * solN[2*I-2]
            + fmin(2*(solN[3*I-1] - solN[3*I-2])/ (x_[I] - x_[I-2]),0.0) * solN[2*I-1] ) ) )

     # potential at right boundary
    residual[3*I-1] = (
        (
        # up boundary condition
        # epsilon[I]*(gI-aI*solN[3*I-1])/bI
        # new up boundary condition just dirichlet
        epsilon[I-1]*(gI - solN[3*I-1])/(x_[I] - x_[I-1])
        # down
        -epsilon[I]*2*(solN[3*I-1]-solN[3*I-2])/(x_[I]-x_[I-2]) )/(x_[I]-x_[I-1])
                + chi2*(solN[I-1]  - solN[2*I-1]))

    return residual

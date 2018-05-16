import numpy as np
cimport numpy as cnp

# Define constants
_EPSILON_VAC = 8.854187817e-12 # original value in [Farad/m],
_ELE_CHARGE = 1.6021766208e-19 # [Coulomb], constant value
_BOLTZMANN_CONST = 1.38064852e-23 # [Joule/Kelvin], constant value 
_AVOGADRO_CONST = 6.022140857e23 # [1/mol]
_GAS_CONST = 8.314459848 # [J/(K*mol)]
_FARADAY_CONST = 96485.3328959 # [C/mol]

def calc_DL_Volt_Cur( cnp.ndarray[ cnp.float64_t, ndim=1] t_, 
                    int I, 
                    cnp.ndarray[ cnp.float64_t, ndim=1] x_,
                    cnp.ndarray[ cnp.float64_t, ndim=2] sol,
                    cnp.ndarray[ cnp.float64_t, ndim=1] gI_,
                    double Dt, 
                    double chi2, 
                    double kC0, 
                    double jC0, 
                    double kCI, 
                    double jCI):
    """
    Calculate DL voltage and current.
    """
    
    cdef:
        cnp.ndarray[ cnp.float64_t, ndim=2] current = np.zeros([2,t_.shape[0]], dtype = np.float64)
        cnp.ndarray[ cnp.float64_t, ndim=2] voltage = np.zeros([2,t_.shape[0]], dtype = np.float64)
        int j = 0
    
    for j in range(1,t_.shape[0]):
                   
        current[0,j] = (-( sol[2*I,j] - sol[2*I,j-1] ) /((x_[1]-x_[0])*Dt*chi2) + (jC0 - kC0*sol[0,j]))     
        current[1,j] = (( gI_[j] - sol[3*I-1,j] - gI_[j-1] + sol[3*I-1,j-1] ) /((x_[I] - x_[I-1])*Dt*chi2) + (kCI*sol[I-1,j] - jCI ))
        
        voltage[0,j] = sol[2*I+int(I/2),j]
        voltage[1,j] = (gI_[j] - sol[2*I+int(I/2),j])
    
    return current, voltage

def sineSeries( cnp.ndarray[ cnp.float64_t, ndim=1] t_,
              double Dt,
              double T0, 
              double U0, 
              double ampl, 
              freq_list, 
              double phi0):
    """
    Frequency: f in [Hz]
    Argument of sine sin(2pi*f*t_) with t_ [s] --> sin(2pi*f*t_*T0) --> Nondim f = f*T0
    
    Input frequency list in multiples of minimum sample frequency
    
    
    """
    cdef:
        int k
        cnp.ndarray[ cnp.float64_t, ndim=1] voltage = np.zeros(t_.size, dtype=np.float64)

    # print
    print("Min Freq, Frequency Resolution Df [Hz]: ", 1/(t_.size*Dt*T0))
    print("Min Freq, Frequency Resolution Df [None]: ", 1/(t_.size*Dt))
    print("Maximal Frequency, Nyquist [Hz]: ", 1/(T0*2*Dt))
    print("Maximal Frequency, Nyquist [None]: ", 1/(2*Dt))
    
    
    # add threshold
    voltage = voltage + U0/phi0
    
    # min frequency 1/(N*Dt)
    for k in freq_list:
        voltage = voltage + ampl/phi0 * np.sin(2*np.pi*(k)* 1/(t_.size*Dt)* t_)
    
    print("Total Amplitude [mV]: ", (voltage.max()-voltage.min())*phi0*1e3)
    
    return voltage

# calculation of numerical constants
def calcChi1( double phi0,
              double T):
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

def calcChi2( double L, 
              double c0, 
              double phi0, 
              double epsilon_m):
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

def calcBjL( double epsilon_m, 
             double T):
    """
    Calculate Bjerrum length.
    
    Input:
        T = temperature if the system, in Kelvin
        epsilon_m = dielectric constant of the medium, dimensionless
    """
    return _ELE_CHARGE**2 / (4*np.pi*_EPSILON_VAC*epsilon_m*_BOLTZMANN_CONST*T)

def calcDbL( double epsilon_m, 
             double T, 
             double I):
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
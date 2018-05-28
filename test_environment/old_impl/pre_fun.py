import numpy as np

class Setup():
    
    ELE_CHARGE = 1.6021766208e-19 # [Coulomb], constant value
    AVOGADRO_CONST = 6.022140857e23 # [1/mol]
    EPSILON_VAC = 8.854187817e-12 # original value in [Farad/m],
    BOLTZMANN_CONST = 1.38064852e-23 # [Joule/Kelvin], constant value

    
    def __init__( self, I, N, Dt, T, L,
    			 lengthscale, c0_in, DA, DC, D0_in,
    			 epsilon, epsilon_m, testname, model, out_method ):
        
        self.I = I
        self.N = N
        self.Dt = Dt
        self.T = T
        self.L = L
        self.lengthscale = lengthscale
        self.c0 = c0_in * self.AVOGADRO_CONST * 1e3 * ( lengthscale ) ** (-3)
        self.DA = DA
        self.DC = DC
        self.D0 = D0_in * (lengthscale ) ** 2
        self.epsilon = epsilon
        self.epsilon_m = epsilon_m
        self.testname = testname
        self.model = int(model)

        # possible output method:
        #				- full: saves current and full solution vector
        #				- c_only: saves only current, avoid saving whole solution in order to save memory

        self.out_method = out_method
        
        self.phi0 = ( self.BOLTZMANN_CONST * T) / self.ELE_CHARGE
        
        self.chi2 = ( self.c0 * lengthscale * (self.ELE_CHARGE * self.L) ** 2 /
                     (self.epsilon_m * self.EPSILON_VAC * self.BOLTZMANN_CONST * T) )
        
        self.chi1 = 1.0
        
        self.T0 = self.L ** 2 / self.D0

def calcAxis( I ):
    
    # create x axis for simulation
    xi = np.zeros(I+1, dtype = np.float64)
    x_ = np.zeros(I+1, dtype = np.float64)

    for i in range(0,I+1):

        xi[i] = i / I
 
    # creating x axis with functional relation
    #x_ = np.sin( ( np.pi * xi  ) / 2 ) ** 2
    x_ = xi
    
    # cell centers
    centers = np.empty(I, dtype = np.float64 )
    for i in range(I):

        centers[i] = x_[i] + ( x_[i+1] - x_[i] ) / 2
    
    # getting cell volumes
    Dx = np.zeros(I, dtype = np.float64)
    Dx = x_[1:] - x_[:I]

    return Dx, centers


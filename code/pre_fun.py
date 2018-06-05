import numpy as np
import pickle
from datetime import date
from time import clock
from scipy.optimize import newton_krylov
import os

from mid_fun import residual as res, constJac, calcAnodicCurrent, calcCatodicCurrent, time_step_full, time_step_red

class Setup():
    """


    Simulation Types:
        - single_const:

            Constant

        c_only_imp

        full_imp


    """
    ELE_CHARGE = 1.6021766208e-19 # [Coulomb], constant value
    AVOGADRO_CONST = 6.022140857e23 # [1/mol]
    EPSILON_VAC = 8.854187817e-12 # original value in [Farad/m],
    BOLTZMANN_CONST = 1.38064852e-23 # [Joule/Kelvin], constant value


    def __init__( self, I, N, Dt, T, L, lengthscale, c0_in, DA, DC, D0_in, 
                 epsilon, epsilon_m, testname, model, sim_method, kA, kC, foxA, foxC, **kwargs ):

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

        # reaction boundary conditions
        self.kA = kA
        self.kC = kC
        self.foxA = foxA
        self.foxC = foxC

        # possible output method:
        #                        - full: saves current and full solution vector
        #                        - c_only: saves only current, avoid saving whole solution in order to save memory

        self.sim_method = sim_method

        self.phi0 = ( self.BOLTZMANN_CONST * T) / self.ELE_CHARGE

        self.chi2 = ( self.c0 * lengthscale * (self.ELE_CHARGE * self.L) ** 2 /
                     (self.epsilon_m * self.EPSILON_VAC * self.BOLTZMANN_CONST * T) )

        self.chi1 = 1.0

        self.T0 = self.L ** 2 / self.D0
        
        # call setter methods
        self.set_ref_flux()
        
        # call setter method for model
        self.set_model( model )
        
        # keyword args
        self.steady_state_tol = kwargs.get('steady_state_tol', 1e-7)
           
#--------- setter methods ------------------------------------------------------------
    def set_ref_flux( self ):
        
        self.f0 = self.L * self.c0 / self.T0
    
    def set_model( self, model ):
        """
        Setter method for model
        Void
        """
        test_attr = ['kA', 'kC', 'foxA', 'foxC']
        logical_list = []
        
        if model == 0:
            
            for element in test_attr:
            
                try:
                
                    if not getattr( self, element) == 0:
                    
                        raise ValueError('Model and boundary conditiosn mismatch!')
                
                except AttributeError:
                    
                    pass
                
            self.model = model
                    
        else:
            
            for element in test_attr:
            
                try:
                
                    if getattr( self, element) == 0:
                    
                        logical_list.append(True)
                
                except AttributeError:
                    
                    self.model = model
            
            if logical_list.__len__() == test_attr.__len__():
                
               raise ValueError('Model and boundary conditions mismatch!')
            
            else:
                
                self.model = model    
    
#----- setter methods for simulation data ------------------------------------------------------------
    def set_current_data( self, current, steady_state = False ):
        """
        Save simulated current data automatically converted to mA / cm^2
        Void
        """
        
        # check input
        if not isinstance(current, np.ndarray):
            
            raise TypeError("Current data is not Numpy array!")
        
        # automatically convert to mA / cm^2
        self.current_data = current * self.f0 * self.ELE_CHARGE * 1e17
        self.simulation_date = str(date.today())

        # extract steady state data
        if steady_state == True:

            self.current_ss = current[:,-1] * self.f0 * self.ELE_CHARGE * 1e17
        
    def set_input_voltage_data( self, voltage, nondim = True ):
        """
        Create pertubation voltage data automatically converted to V
        Void
        """
        
        # check input        
        if isinstance( voltage, np.ndarray ):
            
            if nondim == True:

                self.input_voltage_data = voltage / self.phi0
            
            else:

                self.input_voltage_data = voltage

        elif isinstance( voltage, float ):
            """
            Works for single_const sim method
            """
            # check sim_method
            if self.sim_method == 'single_const':

                # allocate
                self.input_voltage_data = np.zeros( self.N, dtype = np.float64)

                if nondim == True:
                    
                    self.input_voltage_data += voltage / self.phi0

                else:

                    self.input_voltage_data += voltage

            else:
                
                raise ValueError('Voltage input and sim method mismatch!')
    
        else:
            
            raise TypeError("Wrong data type for voltage input!")
            
        
        
    def set_sol_data( self, sol, steady_state = False ):
        """
        Save solution matrix. Automatic convert to M and mV
        Void
        """
        
        # check input
        if not isinstance(sol, np.ndarray):
            
            raise TypeError("Sol data is not Numpy array!")
        
        # convert to potential to mV
        sol[2*self.I:3*self.I,:] *= self.phi0 * 1e3
        
        if steady_state == False:

            self.sol_data = sol

        else:

            self.sol_data = sol[:,-1]

#------- load and save methods -------------------------------------------------------------------
    @classmethod
    def load_setup( cls, filename ):
        """
        Load setup object from .sim file with 'pickle' and return class instance with data
        """
        if not isinstance(filename, str):

                raise TypeError("Optional filename is not a string!")
        
        with open(filename, "rb") as f_id:
            class_instance = pickle.load(f_id)
        
        return class_instance
    
    def save( self, filename_in = None ):
        """
        Save setup object including all data using pickle as obj file.
        Void
        """
        if filename_in is None:
            
            filename = self.testname + ".sim"
            
        else:
        
            # check if filename is a string
            if not isinstance(filename_in, str):

                raise TypeError("Optional filename is not a string!")
            
            else:
                
                filename = filename_in + ".sim"
        
        
        with open(filename, "wb") as file_id:
            pickle.dump( self, file_id)

#--------- simulation related methods -------------------------------------------------------------
    def create_time_axis( self,):
        """
        Create time axis from a instance with given N and Dt
        """
        
        # create time axis
        t = np.zeros(self.N, dtype = np.float64)
        for j in range(0,self.N):

            t[j] = j * self.Dt
            
        return t

    def calcAxis( self, ):

        # create x axis for simulation
        xi = np.zeros(self.I+1, dtype = np.float64)
        x_ = np.zeros(self.I+1, dtype = np.float64)

        for i in range(0,self.I+1):

            xi[i] = i / self.I

        # creating x axis with functional relation
        #x_ = np.sin( ( np.pi * xi  ) / 2 ) ** 2
        x_ = xi

        # cell centers
        centers = np.empty(self.I, dtype = np.float64 )
        
        for i in range(self.I):

            centers[i] = x_[i] + ( x_[i+1] - x_[i] ) / 2

        # getting cell volumes
        Dx = np.zeros(self.I, dtype = np.float64)
        Dx = x_[1:] - x_[:self.I]

        return Dx, centers

    def createVoltage( self, ampl, num = 30 ):

        # create time axis
        t = self.create_time_axis()
        
        voltage = np.zeros(self.N, dtype = np.float64)

        # Sampling and Nyquist Frequency
        f_Ny = np.floor(1.0 / (2.2 * self.Dt)) # Maximal frerquency with secturity 2.2, floor that
        f_s = 1.0 / (self.N * self.Dt)

        singl_ampl = ampl * 1e-3 / self.phi0

        # create factor array
        fac_arr = np.concatenate( (np.array([2, 4, 6, 8]), np.geomspace(10,np.floor(f_Ny / f_s), num = num)) )

        # loop over all multiplicative factors
        for i in range(0, fac_arr.size):

            freq = f_s * int(fac_arr[i]) # calc frequency

            voltage += singl_ampl * np.sin( 2 * np.pi * freq * t ) # add sine to voltage output

        print("Min Freq, Frequency Resolution Df [Hz]: ", 1.0 / (self.N * self.Dt * self.T0))
        print("Min Freq, Frequency Resolution Df [None]: ", 1.0 / (self.N * self.Dt))
        print("Maximal Frequency, Nyquist [Hz]: ", 1.0 / (self.T0 * 2.2 * self.Dt))
        print("Maximal Frequency, Nyquist [None]: ", 1.0 / (2.2 * self.Dt))
        print("Number of Points: ", self.N)
        print("Total Amplitude [mV]: ", np.abs( voltage.max() - voltage.min() ) * self.phi0 *1e3)

        return voltage
    
    def solver( self ):

        M = 1.0

        # initial condition
        sol_initial = np.zeros([3 * self.I], dtype = np.float64)
        sol_initial[0:2*self.I] = 1.0 / 2.0

        # axis
        Dx, centers = self.calcAxis()

        current = np.zeros([2, self.N], dtype = np.float64)
        epsilon_vec = np.ones(self.I + 1, dtype = np.float64) * self.epsilon
        DC_vec = np.ones(self.I + 1, dtype = np.float64) * self.DC
        DA_vec = np.ones(self.I + 1, dtype = np.float64) * self.DA

        # calculate jacobian and invert it for the first points
        Jacinv = constJac(
                      self.I,
                      self.Dt,
                      Dx,
                      DC_vec,
                      DA_vec,
                      epsilon_vec,
                      self.chi2,
                      M
                      )
    
        if self.sim_method == 'single_const':

            # create voltage
            phiC = self.input_voltage_data

            # allocate output vectors
            sol = np.zeros([3 * self.I, self.N], dtype = np.float64)

            sol[:,0] = sol_initial

            print("Start 'single_const' Simulation")

            sol, current, j = time_step_full( self.N,
                    self.I,
                    Dx,
                    sol,
                    self.chi1,
                    self.chi2,
                    DC_vec,
                    DA_vec,
                    self.Dt,
                    self.kA,
                    self.kC,
                    self.foxA,
                    self.foxC,
                    phiC,
                    epsilon_vec,
                    self.model,
                    Jacinv,
                    current,
                    self.steady_state_tol
                     )

            print("Save Results")

            if j<self.N-1:
                
                self.N = j+1


            # set results in setup obj
            self.norm = np.linalg.norm(sol, ord = 2, axis = 0)
            self.set_input_voltage_data( phiC[:j+1], nondim = False  )
            self.set_sol_data( sol[:,:j+1], steady_state = True )
            self.set_current_data( current[:,:j+1], steady_state = True )
           # fehler bei der calculation
           # self.total_concentration = np.tensordot(sol,Dx, axes = ([0], [0]) )

            # save results
            self.save()


        elif self.sim_method == "full_imp":
            phiC, t = createVoltage( setup, 0.2 )
            # allocate output vectors

            sol = np.zeros([3 * self.I, self.N], dtype = np.float64)

            sol[:,0] = sol_initial

            print("Start 'full' Simulation")

            for j in range(1, self.N):

                if j <= 20:
                    print("Time Step: ", j)
                elif j > 20 and np.mod(j,100) == 0:
                    print("Time Step: ", j)

                if j < 5:

                    t_step_method = 1

                    sol2 = sol[:,j-1]

                else:

                    t_step_method = 2

                    sol2 = sol[:,j-2]

                sol[:,j] = newton_krylov( lambda y: res(
                                            self.I,
                                            Dx,
                                            y,
                                            sol[:,j-1], # sol1 (j-1)
                                            sol2, # sol2 (j-2) for BDF2 only
                                            self.chi1,
                                            self.chi2,
                                            DC_vec,
                                            DA_vec,
                                            self.Dt,
                                            self.kA,
                                            self.kC,
                                            self.foxA,
                                            self.foxC,
                                            phiC[j],
                                            epsilon_vec,
                                            self.model,
                                            t_step_method,
                                            M
                                            ),
                                            sol[:,j-1],
                                            inner_M = Jacinv,
                                            method = "lgmres",
                                            verbose = 0,
                                            maxiter = 100)

                # start current calculation if j >= 2, when the system reacts properly
                if j >= 2:

                    # anodic current
                    current[0,j] = calcAnodicCurrent( sol[2*I,j], sol[2*I,j-1], sol[2*I,j-2], self.Dt, Dx[0], self.chi2 )

                    # catodic current
                    current[1,j] = calcCatodicCurrent( sol[3*I-1,j], sol[3*I-1,j-1], sol[3*I-1,j-2],
                                        phiC[j], phiC[j-1], phiC[j-2], self.Dt, Dx[I-1], self.chi2 )

            # save results
            print("Save Results")
            np.save(setup.testname + "_sol.npy", sol)
            np.save(setup.testname + "_current.npy", current)

            # current only output method
        if self.sim_method == "c_only_imp":

            print('Start Impedance - current only - Simulation')

            # create impedance voltage
            phiC = self.createVoltage( 0.25, num = 60 )

            current = time_step_red( self.N,
                    self.I,
                    Dx,
                    sol_initial,
                    self.chi1,
                    self.chi2,
                    DC_vec,
                    DA_vec,
                    self.Dt,
                    self.kA,
                    self.kC,
                    self.foxA,
                    self.foxC,
                    phiC,
                    epsilon_vec,
                    self.model,
                    Jacinv,
                    current,
                    self.steady_state_tol
                     )

            self.set_current_data( current )
            self.set_input_voltage_data( phiC )

            # save results
            print("Save Results")
            self.save()

# end --------------------------------------------------------------------------------------------------------


















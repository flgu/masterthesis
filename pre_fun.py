import numpy as np
import pickle
from datetime import date
from time import clock
from scipy.optimize import newton_krylov
import os

from mid_fun import residual as res, constJac, calcAnodicCurrent, calcCatodicCurrent

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
    def set_current_data( self, current ):
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
            
        
        
    def set_sol_data( self, sol ):
        """
        Save solution matrix. Automatic convert to M and mV
        Void
        """
        
        # check input
        if not isinstance(sol, np.ndarray):
            
            raise TypeError("Sol data is not Numpy array!")
        
        # convert to potential to mV
        sol[2*self.I:3*self.I,:] *= self.phi0 * 1e3
        
        self.sol_data = sol

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
        t = np.zeros(N, dtype = np.float64)
        for j in range(0,N):

            t[j] = j * Dt
            
        return t,
    
    



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

def createVoltage( setup, ampl ):

    N = setup.N
    Dt = setup.Dt
    phi0 = setup.phi0
    T0 = setup.T0

    # create time axis
    t = np.zeros(N, dtype = np.float64)
    for j in range(0,N):

        t[j] = j * Dt


    voltage = np.zeros(N, dtype = np.float64)

    # Sampling and Nyquist Frequency
    f_Ny = np.floor(1.0 / (2.2 * Dt)) # Maximal frerquency with secturity 2.2, floor that
    f_s = 1.0 / (N * Dt)

    singl_ampl = ampl * 1e-3 / setup.phi0

    # create factor array
    fac_arr = np.concatenate( (np.array([2, 4, 6, 8]), np.geomspace(10,np.floor(f_Ny / f_s), num = 60)) )

    # loop over all multiplicative factors
    for i in range(0, fac_arr.size):

        freq = f_s * int(fac_arr[i]) # calc frequency

        voltage += singl_ampl * np.sin( 2 * np.pi * freq * t ) # add sine to voltage output

    print("Min Freq, Frequency Resolution Df [Hz]: ", 1.0 / (N * Dt * T0))
    print("Min Freq, Frequency Resolution Df [None]: ", 1.0 / (N * Dt))
    print("Maximal Frequency, Nyquist [Hz]: ", 1.0 / (T0 * 2.2 * Dt))
    print("Maximal Frequency, Nyquist [None]: ", 1.0 / (2.2 * Dt))
    print("Number of Points: ", N)
    print("Total Amplitude [mV]: ", np.abs( voltage.max() - voltage.min() ) * phi0 *1e3)

    return voltage, t

def solver( setup ):

    I = setup.I
    N = setup.N
    Dt = setup.Dt
    M = 1.0

    # initial condition
    sol_initial = np.zeros([3*I], dtype = np.float64)
    sol_initial[0:2*I] = 1.0 / 2.0

    # reactin boundary conditions
    kA = setup.kA
    kC = setup.kC
    foxA = setup.foxA
    foxC = setup.foxC

    # axis
    Dx, centers = calcAxis( I )

    current = np.zeros([2, N], dtype = np.float64)
    epsilon_vec = np.ones(I + 1, dtype = np.float64) * setup.epsilon
    DC_vec = np.ones(I + 1, dtype = np.float64) * setup.DC
    DA_vec = np.ones(I + 1, dtype = np.float64) * setup.DA

    # calculate jacobian and invert it for the first points
    Jac = constJac(
                  I,
                  Dt,
                  Dx,
                  DC_vec,
                  DA_vec,
                  epsilon_vec,
                  setup.chi2,
                  M
                  )

    Jacinv = np.linalg.inv(Jac)

    
    # delete Jacobian - only inverse J is needed
    del Jac

    if setup.sim_method == 'single_const':

        # create voltage
        phiC = setup.input_voltage_data

        # allocate output vectors
        sol = np.zeros([3 * I, N], dtype = np.float64)

        sol[:,0] = sol_initial

        print("Start 'single_const' Simulation")

        for j in range(1,N):

            if j <= 20:
                print("Time Step: ", j)
            elif j > 20 and np.mod(j,500) == 0:
                print("Time Step: ", j)

            if j < 5:

                t_step_method = 1

                sol2 = sol[:,j-1]

            else:

                t_step_method = 2

                sol2 = sol[:,j-2]

            sol[:,j] = newton_krylov( lambda y: res(
                                        I,
                                        Dx,
                                        y,
                                        sol[:,j-1], # sol1 (j-1)
                                        sol2, # sol2 (j-2) for BDF2 only
                                        setup.chi1,
                                        setup.chi2,
                                        DC_vec,
                                        DA_vec,
                                        Dt,
                                        kA,
                                        kC,
                                        foxA,
                                        foxC,
                                        phiC[j],
                                        epsilon_vec,
                                        setup.model,
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
                current[0,j] = calcAnodicCurrent( sol[2*I,j], sol[2*I,j-1], sol[2*I,j-2], Dt, Dx[0], setup.chi2,
                                setup.model, kA, foxA, sol[0,j] )

                # catodic current
                current[1,j] = calcCatodicCurrent( sol[3*I-1,j], sol[3*I-1,j-1], sol[3*I-1,j-2],
                                    phiC[j], phiC[j-1], phiC[j-2], Dt, Dx[I-1], setup.chi2,
                                    setup.model, kC, foxC, sol[I-1,j] )

            # check convergence to steady state
            if j > 50 and np.linalg.norm(np.subtract(sol[0:I,j], sol[0:I,j-1])) < setup.steady_state_tol:
                print("Steady State reached")
                break

        print("Save Results")

        if j<N-1:

            setup.set_input_voltage_data( phiC[:j+1], nondim = False  )
            setup.N = j+1

        # set results in setup obj
        setup.set_sol_data( sol[:,:j+1] )
        setup.set_current_data( current[:,:j+1] )

        # save results
        
        #np.save(setup.testname + "_sol.npy", sol)
        #np.save(setup.testname + "_current.npy", current)

        





    elif setup.sim_method == "full_imp":
        phiC, t = createVoltage( setup, 0.2 )
        # allocate output vectors

        sol = np.zeros([3 * I, N], dtype = np.float64)

        sol[:,0] = sol_initial

        print("Start 'full' Simulation")

        for j in range(1,N):

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
                                        I,
                                        Dx,
                                        y,
                                        sol[:,j-1], # sol1 (j-1)
                                        sol2, # sol2 (j-2) for BDF2 only
                                        setup.chi1,
                                        setup.chi2,
                                        DC_vec,
                                        DA_vec,
                                        Dt,
                                        kA,
                                        kC,
                                        foxA,
                                        foxC,
                                        phiC[j],
                                        epsilon_vec,
                                        setup.model,
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
                current[0,j] = calcAnodicCurrent( sol[2*I,j], sol[2*I,j-1], sol[2*I,j-2], Dt, Dx[0], setup.chi2 )

                # catodic current
                current[1,j] = calcCatodicCurrent( sol[3*I-1,j], sol[3*I-1,j-1], sol[3*I-1,j-2],
                                    phiC[j], phiC[j-1], phiC[j-2], Dt, Dx[I-1], setup.chi2 )

        # save results
        print("Save Results")
        np.save(setup.testname + "_sol.npy", sol)
        np.save(setup.testname + "_current.npy", current)

        # current only output method
    if setup.sim_method == "c_only_imp":

        # allocate output vectors

        sol1 = np.zeros([3 * I], dtype = np.float64)
        sol2 = np.zeros([3 * I], dtype = np.float64)
        sol3 = np.zeros([3 * I], dtype = np.float64)

        sol2 = sol_initial

        print("Start 'c_only' Simulation")

        for j in range(1,N):

            if j <= 20:
                print("Time Step: ", j)
            elif j < 100 and j > 20 and np.mod(j,10) == 0:
                print("Time Step: ", j)
            elif j > 20 and np.mod(j,5000) == 0:
                print("Time Step: ", j)

            if j < 5:

                t_step_method = 1

            else:

                t_step_method = 2

            sol1 = newton_krylov( lambda y: res(
                                        I,
                                        Dx,
                                        y,
                                        sol2, # sol2 (j-1)
                                        sol3, # sol3 (j-2) for BDF2 only
                                        setup.chi1,
                                        setup.chi2,
                                        DC_vec,
                                        DA_vec,
                                        Dt,
                                        kA,
                                        kC,
                                        foxA,
                                        foxC,
                                        phiC[j],
                                        epsilon_vec,
                                        setup.model,
                                        t_step_method,
                                        M
                                        ),
                                        sol1,
                                        inner_M = Jacinv,
                                        method = "lgmres",
                                        verbose = 0,
                                        maxiter = 100)

            # start current calculation if j >= 2, when the system reacts properly
            if j >= 2:

                # anodic current
                current[0,j] = calcAnodicCurrent( sol1[2*I], sol2[2*I], sol3[2*I], Dt, Dx[0], setup.chi2 )

                # catodic current
                current[1,j] = calcCatodicCurrent( sol1[3*I-1], sol2[3*I-1], sol3[3*I-1],
                                    phiC[j], phiC[j-1], phiC[j-2], Dt, Dx[I-1], setup.chi2 )
            # step solution
            sol3  = sol2
            sol2 = sol1

        setup.set_current_data( current )
        setup.set_input_voltage_data( phiC )

        # save results
        print("Save Results")
        setup.save()

# end --------------------------------------------------------------------------------------------------------

def interactivePlotting( I, sol_arg, centers, L, plotindex = None, markers = None,
                        styles = None, labels = None, savename = None ):
    """
    Plotting time dependent solutions
    """
    label_flag = 0
    marker_flag = 0
    style_flag = 0
    
    # check input
    if isinstance( sol_arg, np.ndarray):
        # normal or interactive plot mode
        
        if plotindex is None and not sol_arg.shape.__len__() == 1:
            
            raise ValueError('Dimension mismatch, have to give plotindex')
        
        elif sol_arg.shape.__len__() == 1:
            
            sol_plot = [ sol_arg ]
            
            # extraxt ion/pot max and min on yaxis and xaxis
            cmin = sol_arg[0:2*I].min() - sol_arg[0:2*I].min() * 0.05
            cmax = sol_arg[0:2*I].max() + sol_arg[0:2*I].max() * 0.05

            phimin = sol_arg[2*I:3*I].min() - sol_arg[2*I:3*I].min() * 0.05
            phimax = sol_arg[2*I:3*I].max() + sol_arg[2*I:3*I].max() * 0.05
        
        elif isinstance(plotindex, int) and sol_arg.shape.__len__() == 2:
            
            sol_plot = [ sol_arg[:,plotindex] ]
            
            # extraxt ion/pot max and min on yaxis and xaxis
            cmin = sol_arg[0:2*I,:].min() - sol_arg[0:2*I,:].min() * 0.05
            cmax = sol_arg[0:2*I,:].max() + sol_arg[0:2*I,:].max() * 0.05

            phimin = sol_arg[2*I:3*I,:].min() - sol_arg[2*I:3*I,:].min() * 0.05
            phimax = sol_arg[2*I:3*I,:].max() + sol_arg[2*I:3*I,:].max() * 0.05
            
    elif isinstance( sol_arg, list):
        # plot single sol pictures
        
        sol_plot = sol_arg
        
        cmin_tmp = min([ el[0:2*I].min() for el in sol_plot ])
        cmax_tmp = max([ el[0:2*I].max() for el in sol_plot ])
        
        pmin_tmp = min([ el[2*I:3*I].min() for el in sol_plot ])
        pmax_tmp = max([ el[2*I:3*I].max() for el in sol_plot ])
        
        # extraxt ion/pot max and min on yaxis and xaxis
        cmin = cmin_tmp - cmin_tmp * 0.05
        cmax = cmax_tmp +  cmax_tmp * 0.05

        phimin = pmin_tmp - pmin_tmp * 0.05
        phimax = pmax_tmp + pmax_tmp * 0.05
        
    if not markers is None:
    
        if not isinstance(markers, list) or not markers.__len__() == sol_plot.__len__():
            
            raise ValueError('Markers have to be a list with same length as sol_arg!')
            
        else:
            
            marker_flag = 1
    
    if not styles is None:
        
        if not isinstance(styles, list) or not styles.__len__() == sol_plot.__len__():
            
            raise ValueError('Styles have to be a list with same length as sol_arg!')
            
        else:
            
            style_flag = 1
            
    if not labels is None:
        
        if not isinstance(labels, list) and not labels.__len__() == sol_plot.__len__():
            
            raise TypeError('Labels have to be a list with same length as sol_arg!')
        
        else:
            
            label_flag = 1
            
    xmin = - L/30.
    xmax = L + L/30.
    
    fig = plt.figure( dpi = 120 )
    
    # create axes
    ax = fig.add_subplot(2,1,1)
    ax1 = fig.add_subplot(2,1,2, sharex = ax)
    
    # iterate over sol_plot list
    for i in range(0, sol_plot.__len__() ):
        
        # cations
        c_line = ax.plot(centers * L, sol_plot[i][0:I], color = "blue", alpha = 0.7, lw = 1.4,)[0];
        
        # anions
        a_line = ax.plot(centers * L, sol_plot[i][I:2*I], color = "red",alpha = 0.7, lw = 1.4,)[0];
        
        # potential
        p_line = ax1.plot(centers * L, sol_plot[i][2*I:3*I],alpha = 0.7, color = "black", lw = 1.4,)[0];
        
        print(p_line)
        
        if marker_flag == 1:
            
            c_line.set_marker(markers[i])
            c_line.set_markersize(4)
            c_line.set_markevery(10)
            
            a_line.set_marker(markers[i])
            a_line.set_markersize(4)
            a_line.set_markevery(10)
            
            p_line.set_marker(markers[i])
            p_line.set_markersize(4)
            p_line.set_markevery(10)
            
        if style_flag == 1:
            
            c_line.set_linestyle(styles[i])
            a_line.set_linestyle(styles[i])
            p_line.set_linestyle(styles[i])
            
        if label_flag == 1:
            
           # c_line.set_label(labels[i])
            #a_line.set_label(labels[i])
            p_line.set_label(labels[i])
            
    # axis spec 
    
    # labels
    ax.set_ylabel(r"Concentration [M]")
    ax1.set_xlabel(r"x [nm]")
    ax1.set_ylabel(r"Potential [mV]")

    # grids
    ax.grid(b=True, which = 'major', axis = 'both')
    ax1.grid(b=True, which = 'major', axis = 'both')
    
    # axis ticks
    plt.setp(ax.get_xticklabels(), visible = False)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%1.1f"))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%1.1f"))
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 8)
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = 8)
    
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    # fill anode catode
    ax.fill_between( [xmin,0],cmin, cmax, facecolor = "red", alpha = 0.4)
    ax1.fill_between( [xmin,0],phimin, phimax, facecolor = "red", alpha = 0.4)
    
    ax.fill_between( [L,xmax+1],cmin, cmax, facecolor = "blue", alpha = 0.4)
    ax1.fill_between( [L,xmax+1],phimin, phimax, facecolor = "blue", alpha = 0.4)
    
    # annotate anode catode
    ax.text(0.005, 0.56, 'Anode', rotation=90,transform=ax.transAxes, fontsize = 8)
    ax1.text(0.005, 0.56, 'Anode', rotation=90,transform=ax1.transAxes, fontsize = 8)
    
    ax.text(0.975, 0.56, 'Catode', rotation=90,transform=ax.transAxes, fontsize = 8)
    ax1.text(0.975, 0.56, 'Catode', rotation=90,transform=ax1.transAxes, fontsize = 8)
    
    # set ax limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([cmin, cmax])
    ax1.set_ylim([phimin, phimax])
    
    # legend
    handles, labels = ax.get_legend_handles_labels()
    handles1, labels1 = ax1.get_legend_handles_labels()
    
    if label_flag == 1:
        
        ax1.legend(handles + handles1, labels + labels1, loc = 9, bbox_to_anchor = (0.55,-0.15), fancybox = True, ncol = 3 )
    
    fig.tight_layout()
    
    if isinstance(savename, str):
        
        for fmt in ['png', 'pdf']:
            fig.savefig(savename + '.' + fmt, dpi = 300, format = fmt, bbox_inches = 'tight' )
    
    plt.show()
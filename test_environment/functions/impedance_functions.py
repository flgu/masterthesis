import numpy as np
import functions.system_tools as st
from scipy.optimize import minimize
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def createImpedanceVoltage(N, Dt, T0, phi0, U_offset = 0, num = 60):

    # create time axis
    time = np.zeros(N, dtype = np.float64)
    for j in range(0,N):
        time[j] = j*Dt

    # Sampling and Nyquist Frequency
    f_Ny = np.floor(1.0 / (2.2 * Dt)) # Maximal frerquency with secturity 2.2, floor that
    f_s = 1.0 / (N * Dt)

    # create factor array
    fac_arr = np.concatenate( (np.array([2, 4, 6, 8]), np.geomspace(10,np.floor(f_Ny / f_s), num = num)) )

    # minimization of the total amplitude through minimization of single amplitudes
    def calcMultiSine( single_ampl, *args ):

        # extract tuple *args (factor_array, f_sample, time, phi0)
        factor_array  = args[0][0]
        f_sample = args[0][1]
        time = args[0][2]

        # define voltage output vector
        voltage = np.zeros(N, dtype = np.float64)

        # loop over all multiplicative factors
        for i in range(0, factor_array.size):

            freq = f_sample * int(factor_array[i]) # calc frequency

            voltage += single_ampl * np.sin( 2 * np.pi * freq * time ) # add sine to voltage output

        # calc total amplitude
        total_ampl = np.abs( voltage.max() - voltage.min() )

        return np.abs(total_ampl - 1.0 / 2.0 )

    # start value = thermal voltage / 2

    # minimize total amplitude
    print("Start optimizing voltage amplitude")
    singl_ampl_opt = minimize( calcMultiSine, 1e-2, args = [fac_arr, f_s, time], tol = 1e-7,
                              bounds = ((0.0, None),)  )

    # use optimized amplitude and create output
    voltage = np.zeros(N, dtype = np.float64)

    # loop over all multiplicative factors
    for i in range(0, fac_arr.size):

        freq = f_s * int(fac_arr[i]) # calc frequency

        voltage += singl_ampl_opt.x * np.sin( 2 * np.pi * freq * time ) # add sine to voltage output

    # add constants offset
    voltage += U_offset

    print("Min Freq, Frequency Resolution Df [Hz]: ", 1.0 / (N * Dt * T0))
    print("Min Freq, Frequency Resolution Df [None]: ", 1.0 / (N * Dt))
    print("Maximal Frequency, Nyquist [Hz]: ", 1.0 / (T0 * 2.2 * Dt))
    print("Maximal Frequency, Nyquist [None]: ", 1.0 / (2.2 * Dt))
    print("Number of Points: ", N)
    print("Total Amplitude [mV]: ", np.abs( voltage.max() - voltage.min() ) * phi0 *1e3)

    return voltage

def plotTypicalInputVoltage(voltage, phi0, T0, Dt, N, time):
    Fvoltage = np.fft.fft(voltage * phi0, n = N)[1:int(N/2)]
    freq_ax = np.fft.fftfreq(N, d = Dt)[1:int(N/2)]

    fig, (ax, ax1) = plt.subplots(2, 1)

    # plot pertubation voltage
    ax.plot(time * T0, voltage * phi0 * 1e3)

    # set grid and ticks
    ax.grid(b = True, which = "major", axis = "both")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # set labels
    ax.set_ylabel(r"$v(t)$ [mV]")
    ax.set_xlabel(r"t [s]")


    ax1.plot(freq_ax / T0, Fvoltage.imag, color = "red", marker = ".")

    # set grid and ticks
    ax1.grid(b = True, which = "major", axis = "both")
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # set labels
    ax1.set_xlabel(r"$\log\left(f \right)$ [Hz]")
    ax1.set_ylabel(r"$\operatorname{Im}(\mathcal{F}(v))$")

    ax1.set_xscale("log")

    fig.tight_layout()

    fig.savefig("voltage_input.pdf",format = "pdf", dpi = 300)

    plt.show()
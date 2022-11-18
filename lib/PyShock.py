import numpy as np  # numpy ... cause you know .... numbers and stuff
import pandas as pd  # library for data analysis and statistics
import requests  # library to handle requests
from bs4 import BeautifulSoup  # library to parse HTML documents

# Constants ------------------------------------------------------------------------------------------------------------
mu0 = 4 * np.pi * 1.00000000055 * pow(10, -7)  # H/m
epsilon0 = 8.8541878128 * pow(10, -12)  # electric constant (F.m-1)
c = 299792458  # metres per second (m/s)
e = 1.602176634 * pow(10, -19)  # the elementary electric charge (C)
h = 6.62607015 * pow(10, -34)  # the plank constant (J.s)
N_a = 6.02214076 * pow(10, 23)  # avogadro constant (mol-1)


def BoltzmannConstant(*args):  # the Boltzmann constant in its various forms
    if args[0] == "J/K":
        return 1.380649 * pow(10, -23)
    if args[0] == "eV/K":
        return 8.617333262 * pow(10, -5)
    if args[0] == "Hz/K":
        return 2.083661912 * pow(10, 10)
    if args[0] == "erg/K":
        return 1.380649 * pow(10, -16)
    if args[0] == "cal/K":
        return 3.297623483 * pow(10, -24)
    if args[0] == "cm-1/K":
        return 0.695034800
    if args[0] == "Eh/K":
        return 3.166811563 * pow(10, -6)


# Unit Converter -------------------------------------------------------------------------------------------------------

def scaleConverter(value: float, scale: str, new_scale: str) -> float:
    units = {'G': 9, 'M': 6, 'k': 3, 'h': 2, 'da': 1, '': 0, 'd': -1, 'c': -2, 'm': -3, 'μ': -6, 'n': -9}
    # G : Giga , M: Mega, k: kilo, h: hecto, da: deca, '': default, d: deci, c: centi, m: milli, μ: micro, n: nano
    gap = units[scale] - units[new_scale]
    return value * 10 ** gap


# ----------------------------------------------------------------------------------------------------------------------

class Resistance:

    # rho: the electrical resistivity of the material, measured in ohm-metres
    # length: length of the conductor, measured in metres
    # area: cross-sectional area of the conductor, measured in square metres

    def __init__(self):
        self.init = self

    @staticmethod
    def createResistance(rho, length, area, *args):
        """The electrical resistance of an object is a measure of its opposition to the flow of electric current.
        Its reciprocal quantity is electrical conductance, measuring the ease with which an electric current passes.
        Electrical resistance shares some conceptual parallels with mechanical friction.
        The SI unit of electrical resistance is the ohm (Ω) - From Wikipedia

        @param rho: the electrical resistivity ρ
        @param length: the length of the specimen
        @param area: is the cross-sectional area of the specimen
        @param args: "Help" for more infor on the resistivity of materials
        @return: is the electrical resistance of a uniform specimen of the material
        """

        # Scraping the wikipedia article to gather the most conductive material for the "Help argument"
        if not args:
            return rho * length / area
        if args[0] == "Help":
            wikiUrl = "https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity"
            table_class = "wikitable"
            response = requests.get(wikiUrl)
            # parse data from the html into a beautifulsoup object
            soup = BeautifulSoup(response.text, 'html.parser')
            resistivityTable = soup.find_all('table', {"class": "wikitable", "class": "sortable"})
            df = pd.read_html(str(resistivityTable))
            # convert list to dataframe
            df = pd.DataFrame(df[0])
            # drop the unwanted columns
            data = df.drop(["Conductivity, σ, at 20\xa0°C (S/m)", "Temperature coefficient[c] (K−1)", "Reference"],
                           axis=1)
            # rename columns for ease
            # rename columns for ease
            data = data.rename(columns={"Resistivity, ρ, at 20\xa0°C (Ω·m)": "Resistivity (Ω·m)"})
            print("This table shows the resistivity (rho), conductivity and temperature coefficient \n of various "
                  "materials at 20 °C (68 °F; 293 K). -From Wikipedia")
            print("----------------------------------------------------------------------")
            print(data.head(n=10))
            print("----------------------------------------------------------------------")

            return 0
        else:
            raise ValueError("The argument can only be Help")

    @staticmethod
    def steinhartHartCoefficients(R1, R2, R3, T1, T2, T3):
        A = [[1, np.log(R1), pow(np.log(R1), 3)],
             [1, np.log(R2), pow(np.log(R2), 3)],
             [1, np.log(R3), pow(np.log(R3), 3)]]

        B = [1 / T1, 1 / T2, 1 / T3]

        return np.linalg.solve(A, B)

    @staticmethod
    def steinhartHartResistance(A=2.108508173e-3, B=0.7979204727e-4, C=6.535076315e-7, T=25 + 273.15, *args):
        """The equation is often used to derive a precise temperature of a thermistor, since it provides a closer
        approximation to actual temperature than simpler equations, and is useful over
        the entire working temperature range of the sensor. Steinhart–Hart coefficients are usually published
        by thermistor manufacturers. - From Wikipedia

        @param A: Steinhart–Hart coefficients, which vary depending on the type and model of thermistor and the
        temperature range of interest.
        @param B: Steinhart–Hart coefficients, which vary depending on the type and model of thermistor and the
        temperature range of interest.
        @param C: Steinhart–Hart coefficients, which vary depending on the type and model of thermistor and the
        temperature range of interest.
        @param T: the temperature (in kelvins),
        @return:  Resistance of a semiconductor at a given temperature (K).
        """

        if not args:
            x = 1 / C * (A - 1 / T)  # calculation of the x coefficient
            y = np.sqrt(pow(B / (3 * C), 3) + pow(x, 2) / 4)  # calculation of the y coefficient
            return np.exp(np.cbrt(y - x / 2) - np.cbrt(y + x / 2))
        elif args[0] == "Plot":
            T = np.arrange(0, 100, 1)
            x = 1 / C * (A - 1 / T)  # calculation of the x coefficient
            y = np.sqrt(pow(B / (3 * C), 3) + pow(x, 2) / 4)  # calculation of the y coefficient
            R = np.exp(np.cbrt(y - x / 2) - np.cbrt(y + x / 2))
            return R

    @staticmethod
    def extrinsicSemiconductor(A, T, n):
        return A * np.exp(pow(T, - 1 / n))


class Functions:

    def __init__(self):
        self.init = self

    @staticmethod
    def series(R1, *args):
        # This function takes multiple values (be it R, L, C) and prints out the sum reactance values as if it were
        # connected in series
        Sum = R1
        for R in args:
            Sum = Sum + R
        return Sum

    @staticmethod
    def parallel(R1, *args):
        # This function takes multiple values (be it R, L, C) and prints out the sum reactance values as if it were
        # connected in parallel.
        # Parameters ---
        # R1: Reactance or Resistor values
        # Output: Impedance of series connected components.
        Sum = 1 / R1
        for R in args:
            Sum = Sum + 1 / R
        return Sum


def Conductance(sigma, length, area, *args):
    if not args:
        return sigma * area / length
    if args[0] == "Help":
        wikiUrl = "https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity"
        response = requests.get(wikiUrl)
        # parse data from the html into a beautifulsoup object
        soup = BeautifulSoup(response.text, 'html.parser')
        resistivityTable = soup.find_all('table', {"class": "wikitable", "class": "sortable"})
        df = pd.read_html(str(resistivityTable))
        # column_names = [item.get_text() for item in resistivityTable[0].find_all('th')]
        # convert list to dataframe
        df = pd.DataFrame(df[0])
        # drop the unwanted columns
        data = df.drop(["Resistivity, ρ, at 20\xa0°C (Ω·m)", "Temperature coefficient[c] (K−1)", "Reference"],
                       axis=1)
        # rename columns for ease
        # rename columns for ease
        data = data.rename(columns={"Conductivity, σ, at 20\xa0°C (S/m)": "Conductivity (S/m)"})
        print("This table shows the resistivity (rho), conductivity and temperature coefficient \n of various "
              "materials at 20 °C (68 °F; 293 K). -From Wikipedia")
        print("----------------------------------------------------------------------")
        print(data.head(n=10))
        print("----------------------------------------------------------------------")
        return 0
    else:
        raise ValueError("The argument can only be Help")


class Inductance:
    def __int__(self):
        self.init = self

    @staticmethod
    def Energy(L, i):
        """ Calculates the energy stored in an inductor

        :param L: magnetic inductance (H)
        :param i: electric current (I)
        :return: stored energy in the inductor (J)
        """
        return L * pow(i, 2) / 2  # calculates the energy stored in an inductor.

    @staticmethod
    def QualityFactor(f, L, R):
        omega = 2 * np.pi * f
        return omega * L / R

    @staticmethod
    def CornerFrequency(R, L):
        return R / (2 * np.pi * L)

    @staticmethod
    def solenoid(N, Area, length):
        return mu0 * pow(N, 2) * Area / length

    @staticmethod
    def singleLayerSolenoid(N, D, length):
        """ The well-known Wheeler's approximation formula for current-sheet model air-core coil.[1, 2]
        This formula gives an error no more than 1% when l is bigger than 0.4 * D.

        Reference
        -------------
        [1] Wheeler, Harold A. (September 1942). "Formulas for the skin effect". Proceedings of the I.R.E.: 412–424
        [2] Wheeler, Harold A. (October 1928). "Simple inductance formulas for radio coils".
            Proceedings of the I.R.E.: 1398–1400.
        @param N: number of turns
        @param D: diameter in (cm)
        @param length:  length in  (cm)
        @return: inductance in (muH)
        """
        return pow(N, 2) * pow(D, 2) / (45 * D + 100 * length)

    @staticmethod
    def coaxialCable(length, b, a):
        # b: Outer conductors inside radius
        # a: Inner conductors radius
        # length: Length
        # mu0: permeability of free space
        return mu0 / (2 * np.pi) * length * np.ln(b / a)

    @staticmethod
    def straightWireConductor(length, radius, mu, rho, omega):
        """ Calculation of the straight wire conductor inductance.

        @param length: length of the cylinder
        @param radius: radius of the cylinder
        @param mu: permeability of the conductor
        @param rho: resistivity of the conductor
        @param omega: phase rate
        @return: inductance value
        """
        A = np.log(length / radius + np.sqrt(pow(length / radius, 2) + 1))
        B = 1 / (radius / length + np.sqrt(1 + pow(radius / length, 2)))
        C = 1 / (4 + radius * np.sqrt(2 / rho * omega * mu))

        return mu0 / (2 * np.pi) * length * (A - B + C)

    @staticmethod
    def reactance(omega, L):
        # omega: the angular frequency
        # L: inductance of element

        return omega * L


class Capacitance:

    def __init__(self):
        self.init = self

    @staticmethod
    def parallelPlate(epsilonr, Area, distance):
        # epsilonr: relative permittivity
        # epsilon0: the electric constant
        # d: separation between the plates
        # Area: area of the two plates,

        return epsilon0 * epsilonr * Area / distance

    @staticmethod
    def concentricCylinders(epsilonr, length, R1, R2):
        epsilon = epsilonr * epsilon0

        return 2 * np.pi * epsilon * length / np.ln(R2 / R1)

    @staticmethod
    def eccentricCylinders(epsilonr, length, R1, R2, distance):
        epsilon = epsilonr * epsilon0

        return 2 * np.pi * epsilon * length / np.arcosh((pow(R1, 2) + pow(R2, 2) - pow(distance, 2)) / (2 * R1 * R2))

    @staticmethod
    def pairOfParallelWires(epsilonr, length, distance, wireRadius):
        epsilon = epsilonr * epsilon0

        return np.pi * epsilon * length / np.arcosh(distance / (2 * wireRadius))


class ElectricCircuits:

    def __int__(self):
        self.init = self

    @staticmethod
    def rlcCircuitSeries(R, L, C, f):
        """ Analysis of RLC Circuit

        :param R: Electric Resistance (Ohm)
        :param L: Magnetic Inductance (H)
        :param C: Electric Capacitance (F)
        :param f: frequency
        :return: power factor, quality factor, resonance angular frequency, impedance
        """
        pf = R / np.sqrt(pow(R, 2) + pow(2 * np.i * f * L - 1 / (2 * np.pi * f * C), 2))  # power factor
        omega = 1 / np.sqrt(L * C)  # resonance angular frequency
        Q = np.sqrt(L / C) / R  # quality factor
        Z = np.sqrt(pow(R, 2) + pow(omega * L - 1 / (omega * C), 2))

        return pf, Q, omega, Z


def rc_circuit(V_in, t, R, C):
    return V_in * (1 - np.exp(-t / (R * C)))


def lorentzForce(q, E, v, B, theta):
    """ Calculates the lorentz force acting on the particle.

    :param q: Total charge acting on the particle (C)
    :param E: The electric field acting on the particle (V/m or N/C)
    :param v: Velocity of the charged particle (m/s)
    :param B: The magnetic field acting on the particle (T)
    :param theta: The angle between the velocity of the particle and the magnetic field vector (rad)
    :return: Lorentz force (N)
    """
    return q * E + q * v * B * np.sin(theta)


def wheatstone(Vin, R1, R2, R3):
    Rx = R2 / R1 * R3
    Vout = Vin * (Rx / (R3 + Rx) - R2 / (R1 + R2))

    return Rx, Vout


def skinDepth(rho, f, mur):
    """ Calculates the skin depth where the majority of the AC current flows.

    :param rho: the resistivity of the conductor
    :param f: frequency of the AC current
    :param mur: the relative permeability of the conductor
    :return: the depth of the conductor where the current flows.
    """
    return np.sqrt(rho / (np.pi * f * mur * mu0))


def timer555(C, R1, R2, *args):
    f = 1.44 / ((R1 + 2 * R2) * C)  # frequency (1/s)
    T = 1 / f  # Period (s)
    T1 = 0.694 * (R1 + R2) * C  # High Time (s)
    T0 = 0.694 * R2 * C  # Low Time (s)
    MSR = T1 / T0  # Calculate the Mark Space Ratio
    Duty = T1 / T  # Calculate the Duty Cycle

    if not args:
        return f, T, T1, T0, MSR, Duty
    elif args[0] == "Print":
        print("The frequency is", f, "Hz")
        print("The period is", T), "s"
        print("The high time is", T1, "s")
        print("The low time is", T0, "s")
        print("The Mark Space Ratio is", MSR)
        print("The Duty Cycle is", Duty)
    else:
        raise ValueError("The argument can only be Print")


def buckConverter(V_out, V_in, I_out, f_sw, V_f, R_DS_on, *args):
    """Calculator to print out the characteristic values of a Buck (Step Down) Calculator

    :param V_out: The output voltage of the buck converter (V)
    :param V_in: The input voltage of the buck converter (V)
    :param I_out: The nominal output current (A)
    :param f_sw: The switching frequency of the converter (kHz)
    :param V_f: The diode voltage drop (V)
    :param R_DS_on: The switch on resistance (Ohm)
    """
    k = 0.1
    k_ripple = 0.001

    P_out = V_out * I_out  # the output power of the converter (W)
    I_out_min = k * I_out  # the minimum output current (I)
    V_pp_ripple = k_ripple * V_out  # the maximum allowable peak-to-peak ripple (V)
    V_R_DS_on = R_DS_on * I_out  # the voltage drop across the on switch (Ohm)

    Duty = (V_out + V_f) / (V_in - V_R_DS_on)  # Duty cycle (must be between 0 and 1)
    T = (1 / f_sw * 1000) / pow(10, 6)  # switching period (s)
    t_on = Duty * T  # on-time of the switch (s)

    L_min = (V_in - V_out - V_R_DS_on) * t_on / (2 * I_out_min)  # minimum inductor values (H)
    E_min = (L_min * pow(I_out + I_out_min, 2)) / 2  # Stored energy in the inductor (J)

    I_pp_ripple = I_out_min * 2  # peak-to-peak ripple current (A)
    I_peak = I_out + I_out_min  # peak switch current (A)
    I_RMS = np.sqrt(
        (V_out + V_f) / (V_in - V_R_DS_on) * (pow(I_peak, 2) - (I_peak * I_pp_ripple) + (pow(I_pp_ripple, 2) / 3)))

    P_cond = R_DS_on * pow(I_RMS, 2)  # the conduction losses occurring on the switch during on state (W)

    V_DC_block = V_in  # DC blocking voltage (V)
    I_avg = I_out * (1 - Duty)  # Average rectified output current (A)
    V_DS_min = (V_in + V_f) + 5  # minimum rated drain-to-source voltage (V)

    I_RMS_C = I_pp_ripple / np.sqrt(12)  # Output capacitor RMS ripple current
    C_out_min = (I_pp_ripple * T) / (8 * V_pp_ripple)

    if not args:
        return Duty
    elif args[0] == "Print":
        print("The switching period is:", T * pow(10, 6), "mus")
        print("The on-time of the switch is:", t_on * pow(10, 6), "mus")
        print("The minimum inductor value is:", L_min * pow(10, 6), "muH")
        print("The peak to peak current is:", I_pp_ripple, "A")
        print("The peak switching current is:", I_peak, "A")
        print("The RMS current is:", I_RMS, "A")
        print("The average rectified output current is: ", I_avg, "A")
        print("The Duty Cycle is:", Duty)
        print("The output capacitor RMS ripple current is:", I_RMS_C, "A")
        print("The minimum output capacitance is:", C_out_min * pow(10, 6), " muF")
    else:
        raise ValueError("The argument can only be Print")


def diodeEquation(I_0, V, T):
    """Calculated the current passing through a diode during operation.

    :param I_0: "dark saturation current", the diode leakage current density in the absence of light. (A)
    :param V: applied voltage across the terminals of the diode (V)
    :param T: absolute temperature (K)
    :return: the net current flowing through the diode (I).
    """
    return I_0 * (np.exp(1 * V / (BoltzmannConstant("J/K") * T)) - 1)


def deltaConnection(array, Z_total):
    return 1


def mmfSum(n, t, phase):
    total = 0
    k = 0
    while k <= n:
        total = total + 1 / (2 * k + 1) * pow(-1, k) * np.cos((2 * k + 1) * (np.pi * t - phase))
        k += 1
    return total * 100


class DCMotor(object):

    def __init__(self):
        self.init = self

    @staticmethod
    def AvgConductorEMF(N_p, Phi_m, omega_a):
        """ Calculates the average conductor EMF experienced by the DC drive

        @param N_p: the number of pole pairs.
        @param Phi_m: flux per pole.
        @param omega_a: speed of the armature.
        @return: average conductor EMF.
        """
        return N_p * omega_a * Phi_m / np.pi

    @staticmethod
    def ArmatureTerminalVoltage(N_p, Phi_m, omega_a, z, a):
        return N_p * omega_a * Phi_m / np.pi * z / (2 * a)

    @staticmethod
    def ArmatureTangentialForce(I_a, a, l_a, B_sigma_x):
        return I_a / (2 * a) * l_a * B_sigma_x

    @staticmethod
    def AverageArmatureTangentialForce(I_a, a, tau_p, Phi_m):
        return I_a / (2 * a) * Phi_m * 1 / tau_p

    @staticmethod
    def ArmatureTangentialTorque(N_p, tau_p, I_a, a, l_a, B_sigma_x):
        return N_p * tau_p / np.pi * I_a / (2 * a) * l_a * B_sigma_x

    @staticmethod
    def AverageArmatureTangentialTorque(N_p, I_a, a, Phi_m):
        return N_p / np.pi * I_a / (2 * a) * Phi_m

    @staticmethod
    def TotalElectromagneticTorque(z, a, N_p, I_a, Phi_m):
        return z / (2 * np.pi * a) * N_p * I_a * Phi_m

    @staticmethod
    def TerminalVoltage(E_a, R_a, I_a, V_b):
        return E_a + R_a * I_a + V_b

    class SeparatelyExcitedDC(object):

        @staticmethod
        def speed(V_a, k, Phi_m, R_a, R_u, T_a):
            return V_a / (k * Phi_m) - (R_a + R_u) / pow(k * Phi_m, 2) * T_a

        @staticmethod
        def noloadspeed(V_a, k, Phi_m):
            return V_a / (k * Phi_m)


class MMF(object):

    @staticmethod
    def spaceFieldCurveDC(B_1, x, tau_p, *args):
        if args[0] == "Fundamental":
            return B_1 * np.cos(x * np.pi / tau_p)
        if args[1] == "I want it all!":
            return 0  # add infinite sum here !

    @staticmethod
    def spaceFieldCurveAC(B_1, x, tau_p, f, t, *args):
        omega = 2 * np.pi * f
        if args[0] == "Fundamental":
            return B_1 * np.cos(x * np.pi / tau_p) * np.cos(omega * t)
        if args[1] == "I want it all!":
            return 0  # add infinite sum here !


def multiPhaseWaveformGeneration(phase, Amplitude, frequency, t):
    diff = 2 * np.pi / phase
    k = 1
    Source = np.zeros((phase, len(t)))
    while k <= phase:
        Source[k - 1] = Amplitude * np.cos(2 * np.pi * frequency * t - k * diff)
        k += 1

    return Source


def ShuntMotorConstruction(P_n, V_n, n):
    C = P_n / n  # usefulness factor (kW * min / m3)

    Cmin = 1.76
    Cmax = 2.65

    C = 2.04 * pow(10, 3)

    eta = 0.865

    # The efficiency of the electromagnetically generated power
    internal_eta = (2 * eta) / (1 + eta)

    # The electromagnetic power
    P_i = P_n / internal_eta

    # Armature diameter
    D_a = 270  # mm
    print("Armature diameter is:", D_a, "mm")

    v_a = np.pi * D_a * n / 60
    print("Rotational Speed is:", v_a, "m/s")

    # Number of pole pairs
    p = 2
    print("Number of pole pairs is:", p)

    # frequency
    f = p * n / 60
    print("Operation frequency is:", f, "Hz")

    # pole step
    tau_p = np.pi * D_a / (2 * p)
    print("Pole step is:", tau_p, "cm")

    a_i = 0.684

    L_i = a_i * tau_p

    P_i = C * pow(D_a, 2) * L_i * n

    m = 4.03

    h_pa = 0.35 * tau_p + m

    control = L_i / tau_p


def telegraphersEquation(R, L, C, G, l, x, t):
    alpha = G / C

    beta = R / L

    c = np.sqrt(1 / (L * C))

    d = (alpha + beta) / 2

    a = 5
    limit = 100
    n = 1

    U = np.zeros((1, len(x)))

    Sum = 0

    while n <= limit:
        omega_n = np.sqrt(pow(n * np.pi * c / l, 2) - 1 / 4 * pow(alpha - beta, 2))

        phi_n = np.arctan(d / omega_n)

        A_n = 2 / (l * np.cos(phi_n)) * np.sin(n * np.pi * a / l)

        u = A_n * np.exp(-d * t) * np.cos(omega_n * t - phi_n) * np.sin(n * np.pi * x / l)

        Sum = Sum + u

        n += 1

    return Sum


def chuaCircuitPlot(alpha=15.395, beta=28, R=- 1.143, C_2=-0.714, u=[1, 1, 1]):
    """Chua's circuit (also known as a Chua circuit) is a simple electronic circuit that exhibits classic chaotic
     behavior. This means roughly that it is a "nonperiodic oscillator"; it produces an
     oscillating waveform that, unlike an ordinary electronic oscillator, never "repeats".
     It was invented in 1983 by Leon O. Chua, who was a visitor at Waseda University in Japan at that time.
     - From Wikipedia

    @param alpha:
    @param beta:
    @param R:
    @param C_2:
    @param u:
    @param t:
    @return:
    """
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def chua():
        x, y, z = u
        # electrical response of the nonlinear resistor
        f_x = C_2 * x + 0.5 * (R - C_2) * (abs(x + 1) - abs(x - 1))
        dudt = [alpha * (y - x - f_x), x - y + z, -beta * y]
        return dudt

    # time values
    t_0 = 0
    dt = 1e-3
    t_final = 300
    t = np.arange(t_0, t_final, dt)

    # initial conditions
    u0 = [0.1, 0, 0]
    # integrate ode system
    sol = odeint(chua, u0, t)

    # 3d-plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2])
    plt.show()


def delta2wye(R_ab, R_ac, R_bc):
    R_a = R_ac * R_ab / (R_ac + R_bc + R_bc)
    R_b = R_ab * R_bc / (R_ac + R_bc + R_bc)
    R_c = R_bc * R_ac / (R_ac + R_bc + R_bc)
    return R_a, R_b, R_c


def wye2delta(R_a, R_b, R_c):
    R_ac = (R_a * R_b + R_b * R_c + R_c * R_a) / R_b
    R_ab = (R_a * R_b + R_b * R_c + R_c * R_a) / R_c
    R_bc = (R_a * R_b + R_b * R_c + R_c * R_a) / R_a
    return R_ac, R_ab, R_bc


def powerGain(P_out, P_in, *args):
    if not args:
        return 10 * np.log10(P_out / P_in)  # dB
    elif args[0] == 'dB':
        return 10 * np.log10(P_out / P_in)  # dB
    elif args[0] == 'nepers':
        return 0.5 * np.log2(P_out / P_in)  # dB
    else:
        raise ValueError("The arguments can either be dB or nepers.")


def stepperMotorResonanceFreq(p, M_h, J_r):
    """When the motor moves a single step it overshoots the final resting point and oscillates round this
    point as it comes to rest. This undesirable ringing is experienced as motor rotor vibration and is more
    pronounced in unloaded motors. An unloaded or under loaded motor may, and often will, stall if the vibration
    experienced is enough to cause loss of synchronisation. Stepper motors have a natural frequency of operation.
    When the excitation frequency matches this resonance the ringing is more pronounced, steps may be missed,
    and stalling is more likely. Motor resonance frequency can be calculated from the formula:

    @param p: Number of pole pairs
    @param M_h: Holding torque (N.m)
    @param J_r: Rotor inertia (kg.m2)
    @return: Motor resonance frequency
    """
    return 100 / (2 * np.pi) * np.sqrt(2 * p * M_h / J_r)

def PhasorPlot(Array):
    from matplotlib import pyplot as plt
    import seaborn

    x_main = 0
    y_main = 0

    PlotArray = np.zeros((len(Array), 4))

    for i in range(len(Array)):
        x = Array[i][0]
        y = Array[i][1]
        x_main = x_main + x
        y_main = y_main + y

    x_main = abs(x_main)
    y_main = abs(y_main)

    for i in range(len(Array) - 1):
        PlotArray[i][2] = Array[i + 1][0]
        PlotArray[i][3] = Array[i + 1][1]

        x_sub_main = 0
        y_sub_main = 0

        j = 0
        while j <= i + 1:
            x = Array[j][0]
            y = Array[j][1]
            x_sub_main = x_sub_main + x
            y_sub_main = y_sub_main + y
            j += 1
        PlotArray[i + 1][0] = x_sub_main
        PlotArray[i + 1][1] = y_sub_main

    PlotArray[-1] = [0, 0, x_main, y_main]

    X, Y, U, V = zip(*PlotArray)
    x = np.linspace(0.2, 10, 100)
    seaborn.set(style='ticks')

    plt.figure()
    plt.ylabel("Imaginary - Axis")
    plt.xlabel("Real - Axis")
    plt.plot(0, 0, 'ok')  # <-- plot a black point at the origin
    ax = plt.gca()
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', color=['r', 'b', 'g'], scale=1)
    ax.set_xlim([- x_main - 5, x_main + 5])
    ax.set_ylim([- y_main - 5, y_main + 5])
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    plt.axvline(0)
    plt.axhline(0)
    seaborn.despine(ax=ax, offset=0)  # the important part here
    plt.draw()
    plt.show()
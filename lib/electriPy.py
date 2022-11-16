import numpy as np

from .__init__ import *


class Resistance:

    # rho: the electrical resistivity of the material, measured in ohm-metres
    # length: length of the conductor, measured in metres
    # area: cross-sectional area of the conductor, measured in square metres

    def __init__(self, rho, length, area):
        self.rho = rho
        self.length = length
        self.area = area

    def createResistance(self):
        return self.rho * self.length / self.area


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


class Conductance:

    # sigma: the electrical conductivity of the material, measured in siemens per meter.
    # length: length of the conductor, measured in metres
    # area: cross-sectional area of the conductor, measured in square metres

    def __init__(self, sigma, length, area):
        self.sigma = sigma
        self.length = length
        self.area = area

    def createConductance(self):
        return self.sigma * self.area / self.length


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
    def solenoid(N, Area, length):
        return c.mu0 * pow(N, 2) * Area / length

    @staticmethod
    def coaxialCable(length, b, a):
        # b: Outer conductors inside radius
        # a: Inner conductors radius
        # length: Length
        # mu0: permeability of free space
        return mu0 / (2 * pi) * length * ln(b / a)

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

        return c.epsilon0 * epsilonr * Area / distance

    @staticmethod
    def concentricCylinders(epsilonr, length, R1, R2):
        epsilon = epsilonr * c.epsilon0

        return 2 * c.pi * epsilon * length / ln(R2 / R1)

    @staticmethod
    def eccentricCylinders(epsilonr, length, R1, R2, distance):
        epsilon = epsilonr * c.epsilon0

        return 2 * c.pi * epsilon * length \
               / arcosh((pow(R1, 2) + pow(R2, 2) - pow(distance, 2)) / (2 * R1 * R2))

    @staticmethod
    def pairOfParallelWires(epsilonr, length, distance, wireRadius):
        epsilon = epsilonr * c.epsilon0

        return c.pi * epsilon * length / arcosh(distance / (2 * wireRadius))


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
        pf = R / root(pow(R, 2) + pow(2 * pi * f * L - 1 / (2 * pi * f * C), 2), 2)  # power factor
        omega = 1 / root(L * C, 2)  # resonance angular frequency
        Q = root(L / C, 2) / R  # quality factor
        Z = root(pow(R, 2) + pow(omega * L - 1 / (omega * C), 2), 2)

        return pf, Q, omega, Z


def rc_circuit(V_in, t, R, C):
    return V_in * (1 - exp(-t / (R * C)))


def lorentzForce(q, E, v, B, theta):
    """ Calculates the lorentz force acting on the particle.

    :param q: Total charge acting on the particle (C)
    :param E: The electric field acting on the particle (V/m or N/C)
    :param v: Velocity of the charged particle (m/s)
    :param B: The magnetic field acting on the particle (T)
    :param theta: The angle between the velocity of the particle and the magnetic field vector (rad)
    :return: Lorentz force (N)
    """
    return q * E + q * v * B * sin(theta)


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
    return root(rho / (c.pi * f * mur * c.mu0), 2)


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
        print("The frequency is", f)
        print("The period is", T)
        print("The high time is", T1)
        print("The low time is", T0)
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
    I_RMS = root(
        (V_out + V_f) / (V_in - V_R_DS_on) * (pow(I_peak, 2) - (I_peak * I_pp_ripple) + (pow(I_pp_ripple, 2) / 3)), 2)

    P_cond = R_DS_on * pow(I_RMS, 2)  # the conduction losses occurring on the switch during on state (W)

    V_DC_block = V_in  # DC blocking voltage (V)
    I_avg = I_out * (1 - Duty)  # Average rectified output current (A)
    V_DS_min = (V_in + V_f) + 5  # minimum rated drain-to-source voltage (V)

    I_RMS_C = I_pp_ripple / root(12, 2)  # Output capacitor RMS ripple current
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
    return I_0 * (exp(1 * V / (BoltzmannConstant().J_per_K() * T)) - 1)


def deltaConnection(array, Z_total):
    return 1


def mmfSum(n, t, phase):
    sum = 0
    k = 0
    while k <= n:
        sum = sum + 1 / (2 * k + 1) * pow(-1, k) * np.cos((2 * k + 1) * (np.pi * t - phase))
        k += 1
    return sum * 100


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
            return 0 # add infinite sum here !
    @staticmethod
    def spaceFieldCurveAC(B_1, x, tau_p, f, t,*args):
        omega = 2 * np.pi * f
        if args[0] == "Fundamental":
            return B_1 * np.cos(x * np.pi / tau_p) * cos(omega * t)
        if args[1] == "I want it all!":
            return 0 # add infinite sum here !


def multiPhaseWaveformGeneration(phase, Amplitude, frequency, t):
    diff = 2 * np.pi / phase
    k = 1
    Source = np.zeros((phase, len(t)))
    while k <= phase:
        Source[k-1] = Amplitude * np.cos(2 * np.pi * frequency * t - k * diff)
        k += 1

    return Source
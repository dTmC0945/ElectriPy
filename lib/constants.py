from .__init__ import *
# Constants

pi = 3.14159265358979323846  # the ratio of a circle's circumference to its diameter
mu0 = 4 * pi * 1.00000000055 * pow(10, -7)  # H/m
epsilon0 = 8.8541878128 * pow(10, -12)  # F / m-1
euler = 2.718281828459045235360287471352
c = 299792458  # metres per second
e = 1.602176634 * pow(10, -19)  # the electric charge carried by a single proton or,


# equivalently, the magnitude of the negative electric charge carried by a single electron (C)


class BoltzmannConstant():
    def __init__(self):
        self.init = self

    @staticmethod
    def J_per_K():
        return 1.380649 * pow(10, -23)

    @staticmethod
    def eV_per_K():
        return 8.617333262 * pow(10, -5)

    @staticmethod
    def Hz_per_K():
        return 2.083661912 * pow(10, 10)

    @staticmethod
    def erg_per_K():
        return 1.380649 * pow(10, -16)


@staticmethod
def convert_SI(val, unit_in, unit_out):
    SI = {'mm': 0.001, 'cm': 0.01, 'm': 1.0, 'km': 1000.}
    return val * SI[unit_in] / SI[unit_out]


pi = 3.14159265358979323846
mu0 = 4 * pi * 1.00000000055 * pow(10, -7)  # H/m
epsilon0 = 8.8541878128 * pow(10, -12)  # F / m-1
euler = 2.718281828459045235360287471352
c = 299792458  # metres per second


@staticmethod
def convert_SI(val, unit_in, unit_out):
    SI = {'mm': 0.001, 'cm': 0.01, 'm': 1.0, 'km': 1000.}
    return val * SI[unit_in] / SI[unit_out]

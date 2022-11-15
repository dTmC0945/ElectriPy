from lib import *
import matplotlib.pyplot as plt
import numpy as np
from mpmath import nsum, inf


t1 = np.arange(0.01, pi/2, 0.02,  dtype=int)


def f(t):
    return nsum(lambda x: (1/(2*x + 1) * cos((2 * x + 1)*t1) * pow(-1, 2*x + 1)), [0, 30], method='shanks')

for i in t1:
    nsum(lambda x: (1 / (2 * x + 1) * cos((2 * x + 1) * t1[i]) * pow(-1, 2 * x + 1)), [0, 50], method='shanks')


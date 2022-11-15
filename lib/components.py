from .__init__ import *


class SinglePhaseACSource(object):
    def __init__(self, Type='AC', Hz=50, A=120, Phase=0):
        self.Type = Type
        self.Hz = Hz
        self.A = A
        self.Phase = Phase
        self.w = pi * 2 * self.Hz

    def source(self, t):
        return self.A * cos(self.w * t + self.Phase)

    def rms(self):
        return self.A / root(2, 2)


class Diode(object):
    def __init__(self, R_on=0.0001, V_f=0.6, I_0=0):
        self.R_on = R_on
        self.V_f = V_f
        self.I_0 = I_0

    def vform(self, array):
        output = zeroMatrix(1, len(array))

        for i in range(array):
            if (array[i] - self.V_f) >= 0:
                output[i] = array[i] - self.V_f

        return output

class Resistor(object):

    def __init__(self, R):
        self.R = R


from __init__ import *


class SinglePhaseACSource(object):
    def __init__(self, Type='AC', Hz=50, A=120, Phase=0):
        self.Type = Type
        self.Hz = Hz
        self.A = A
        self.Phase = Phase
        self.w = pi * 2 * self.Hz

    def Source(self, t):
        return self.A * cos(self.w * t + self.Phase)

    def RMS(self):
        return self.A / root(2, 2)

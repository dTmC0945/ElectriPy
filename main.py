from lib import *
import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0.01, np.pi/4, 0.02)




y = multiPhaseWaveformGeneration(3, 1, 2, x)

y1 = y[0,:]
y2 = y[1,:]
y3 = y[2,:]
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.show()
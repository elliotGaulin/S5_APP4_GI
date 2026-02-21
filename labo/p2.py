import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane

z1 = np.exp(1j * np.pi / 16)
z2 = np.exp(-1j * np.pi / 16)

p1 = 0.95 * z1
p2 = 0.95 * z2

num2 = np.poly([z1, z2])
denum2 = np.poly([p1, p2])

n = np.arange(0, 512)
x = np.sin(n *np.pi/16) + np.sin(n*np.pi/32)

y = signal.lfilter(num2, denum2, x)
plt.figure()
plt.plot(n, x, label='Input signal')
plt.plot(n, y, label='Filtered signal')
plt.title('Input and Filtered Signals')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()
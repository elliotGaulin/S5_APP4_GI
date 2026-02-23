import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane



fe = 48000
nyquist = fe/2
bandwitdh = np.arange(0 , 2500, 1)
fc = 3500
gpass = 0.2
gstop = 40

ord, wn  = signal.buttord(wp = 2500 / nyquist, ws = 3500 / nyquist, gpass = gpass, gstop = gstop)

print("Order of the Butterworth filter:", ord) #18
b, a = signal.butter(ord, 2500 / nyquist, btype='low')
w, h = signal.freqz(b, a)
plt.figure()
ax1 = plt.plot((w * nyquist / np.pi)/np.pi, 20 * np.log10(abs(h)))
plt.axhline(y=-3, color='green', linestyle='--') 
plt.axvline(x=2500, color='red', linestyle='--') 
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid()

# ax2 = plt.twinx()
# ax2.set_ylabel('Phase rad / ech')
# angles = np.unwrap(np.angle(h))
# ax2.plot(w * nyquist / np.pi, angles, 'orange')
print("Coefficients b:", b)
print("Coefficients a:", a)
plt.figure()
zplane(b, a)

plt.show()


# chebyshev type 1
ord, wn  = signal.cheb1ord(wp = 2500 / nyquist, ws = 3500 / nyquist, gpass = gpass, gstop = gstop)
print("Order of the Chebyshev type 1 filter:", ord) #11
b, a = signal.cheby1(ord, gpass, 2500 / nyquist, btype='low')
w, h = signal.freqz(b, a)
plt.figure()
ax1 = plt.plot((w * nyquist / np.pi)/np.pi, 20 * np.log10(abs(h)))
plt.axhline(y=-3, color='green', linestyle='--')
plt.axvline(x=2500, color='red', linestyle='--')
plt.title('Chebyshev type 1 filter frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid()
print("Coefficients b:", b)
print("Coefficients a:", a)
zplane(b, a)

plt.show()

# chebyshev type 2
ord, wn  = signal.cheb2ord(wp = 2500 / nyquist, ws = 3500 / nyquist, gpass = gpass, gstop = gstop)
print("Order of the Chebyshev type 2 filter:", ord) #11
b, a = signal.cheby2(ord, gstop, 2500 / nyquist, btype='low')
w, h = signal.freqz(b, a)
plt.figure()
ax1 = plt.plot((w * nyquist / np.pi)/np.pi, 20 * np.log10(abs(h)))
plt.axhline(y=-3, color='green', linestyle ='--')
plt.axvline(x=2500, color='red', linestyle='--')
plt.title('Chebyshev type 2 filter frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid()
print("Coefficients b:", b)
print("Coefficients a:", a)
zplane(b, a)

plt.show()

# elliptic
ord, wn  = signal.ellipord(wp = 2500 / nyquist, ws = 3500 / nyquist, gpass = gpass, gstop = gstop)
print("Order of the Elliptic filter:", ord) #7
b, a = signal.ellip(ord, gpass, gstop, 2500 / nyquist, btype='low')
w, h = signal.freqz(b, a)
plt.figure()
ax1 = plt.plot((w * nyquist / np.pi)/np.pi, 20 * np.log10(abs(h)))
plt.axhline(y=-3, color='green', linestyle='--')
plt.axvline(x=2500, color='red', linestyle='--')
plt.title('Elliptic filter frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid()
print("Coefficients b:", b)
print("Coefficients a:", a)
zplane(b, a)

plt.show()

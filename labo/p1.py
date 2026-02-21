import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane


poles = [0.95 * np.e**(1j * np.pi / 8), 0.95 * np.e**(-1j * np.pi / 8)]
zeros = [0.8j, -0.8j]

num = np.poly(zeros)
denum = np.poly(poles)

#a)
# zplane(num, denum) 


#b) filtre stable

w, h = signal.freqz(num, denum)
print(f"len = {len(h)}")
print(f"max() = {np.max(w)}")

axes = plt.subplot(1,1,1)
# plt.figure()
axes.plot(w / np.pi, 20 * np.log10(np.abs(h)))

axes.set_title('Frequency response')
axes.set_xlabel('Normalized Frequency (xπ rad/sample)')

axes.set_ylabel('Magnitude (dB)')
axes.grid()

axe2 = axes.twinx()
axe2.plot(w / np.pi, np.angle(h), 'g')
axe2.set_ylabel('Phase (radians)', color='g')
axe2.tick_params(axis='y', labelcolor='g')



#impulsion
imp = np.zeros(512)
imp[0] = 1

# axes1 = plt.subplot(1,1,1)
h_n = signal.lfilter(num, denum, imp)
plt.figure()
plt.stem(h_n)
plt.title('Impulse response')
plt.xlabel('n')
plt.ylabel('h[n]')

H_w = np.fft.rfft(h_n, 512)
w = np.fft.rfftfreq(512, d=1/2)  # le d est obscure
plt.figure()
plt.plot(w, 20 * np.log10(np.abs(H_w)))
plt.title('Frequency response (FFT)')
plt.xlabel('Normalized Frequency (xπ rad/sample)')
plt.ylabel('Magnitude (dB)')

# e) H(w)**-1 = 1/H(w) = denum/num
# f)
h_n_2 = signal.lfilter(denum, num, h_n)
plt.figure()
plt.stem(h_n_2)
plt.title('Impulse response of the filtered impulse response')
plt.xlabel('n')
plt.ylabel('h[n]')

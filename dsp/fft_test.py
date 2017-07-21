import matplotlib.pyplot as plt
import numpy as np


Fs = 48.0  # sampling rate
length = 90
Ts = 1.0/Fs  # sampling interval
t = np.arange(0, length, Ts)  # time vector

noise = np.random.random([len(t)]) * 5
ff = 0.5;   # frequency of the signal
no_signal_length = 60 * 48
y = [0 for i in range(no_signal_length)]
for i in t[:int(Fs*length - no_signal_length)]:
    y.append(np.sin(2*np.pi*ff*i))
y = np.array(y)
y += noise

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

plt.show()

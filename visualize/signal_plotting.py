from scipy import signal

import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np

def plot_waveform_fft(waveform, show=True, output_dir=None):
    plt.figure()
    n = waveform.data.shape[0]

    # Sample spacing
    t = 1.0 / waveform.sampling_frequency
    y = waveform.data
    yf = fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * t), int(n / 2))
    plt.plot(xf, 2.0 / n * np.abs(yf[0:n // 2]))
    plt.grid()
    if show:
        plt.show()

def plot_waveform_periodgram(waveform, show=True, output_dir=None):
    plt.figure()
    f, Pxx_den = signal.periodogram(1E4*waveform.data, waveform.sampling_frequency, return_onesided=True,
                                    scaling='density')
    sorted_indices = np.argsort(f)
    plt.plot(f[sorted_indices]/1E6, 10*np.log10(Pxx_den[sorted_indices]))
    plt.xlim([-20, 20])
    plt.ylim([-50, 20])
    plt.xlabel('frequency [MHz]')
    plt.ylabel('PSD dB[V**2/Hz]')
    plt.grid()
    if show:
        plt.show()
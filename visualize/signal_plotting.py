from scipy import signal

import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np

def plot_waveform_signal(waveform, show=True, output_dir=None):
    plt.figure()
    t = np.linspace(0, waveform.samples/waveform.sampling_frequency, waveform.samples)
    plt.plot(t, waveform.data)
    plt.grid(True)
    if show:
        plt.show()

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

def plot_waveform_periodgram(waveform, show=True, flim=None, output_dir=None):
    plt.figure()
    p, freqs = plt.psd(waveform.data,NFFT=2**16, Fs=waveform.sampling_frequency, noverlap=2**15)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD (dB/Hz))')
    if flim is not None:
        plt.xlim(flim)
    plt.grid(True)
    if show:
        plt.show()
import copy

import numpy as np

import scipy
from scipy.fft import rfft, rfftfreq, ifft, fftfreq, fft


class Waveform(object):
    def __init__(self):
        self.sampling_frequency = None
        self.samples = None
        self.data = None

    @property
    def rfft(self):
        fft = rfft(self.data, )
        fft_freqs = rfftfreq(self.data, self.sampling_frequency)
        return fft, fft_freqs

    @property
    def fft(self):
        fft_data = fft(self.data, self.samples)
        fft_freqs = fftfreq(self.samples, 1/self.sampling_frequency)
        return fft_data, fft_freqs

    def upsample(self, sampling_freq):
        self.data = np.array(scipy.signal.resample_poly(self.data, up=sampling_freq, down=self.sampling_frequency))
        self.sampling_frequency = sampling_freq
        self.samples = self.data.shape[0]

    def downsample(self, sampling_freq):
        new_samples = int(self.samples*sampling_freq/self.sampling_frequency)
        curr_time = np.linspace(0, self.samples/self.sampling_frequency, self.samples)
        new_time = np.linspace(0, self.samples/self.sampling_frequency, new_samples)
        indices = abs(curr_time - new_time).argmin()
        self.data = self.data[indices]
        self.samples = new_samples
        self.sampling_frequency = sampling_freq

    def freq_shift(self, frequency_shift):
        self.data *= np.exp(2j*np.pi*frequency_shift*np.linspace(0, self.samples/self.sampling_frequency, self.samples))

    def bandpass_filter(self, freq_bottom, freq_top):
        fft_data, fft_freqs = self.fft
        fft_data[fft_freqs < freq_bottom] = 0.0
        fft_data[fft_freqs > freq_top] = 0.0
        self.data = ifft(fft_data)

    def convolve(self, waveform):
        if self.sampling_frequency > waveform.sampling_frequency:
            waveform.upsample(self.sampling_frequency)
        elif self.sampling_frequency < waveform.sampling_frequency:
            waveform.downsample(self.sampling_frequency)
        return np.convolve(self.data, waveform.data)

    def correlation(self, waveform):
        return np.correlate(self.data, waveform.data)[0]/self.samples

    def adjust_power(self, power):
        self.data = 10 ** (power / 10) / np.mean(abs(self.data) ** 2) * self.data

    def __add__(self, other):
        wfm = Waveform()
        wfm.sampling_frequency = self.sampling_frequency
        wfm.samples = self.samples
        wfm.data = self.data + other.data
        return wfm

    @staticmethod
    def sin(frequencies, amplitude, time, dt):
        waveform= Waveform()
        waveform.data = np.zeros(time.shape[0])
        waveform.samples = time.shape[0]
        waveform.sampling_frequency = 1.0/dt
        for freq, amp in zip(frequencies, amplitude):
            waveform.data += amp * np.sin(2.0*np.pi*freq * time)
        return waveform

    @staticmethod
    def bpsk(samples, sampling_freq, carrier_freq, chipping_freq, bit_sequence=None):
        dt = 1/sampling_freq
        time = samples*dt
        if bit_sequence is None:
            bits = int((time)*chipping_freq)+1
            bit_sequence = np.random.randint(low=0, high=2, size=bits)
            bit_sequence = np.repeat(bit_sequence,int(samples/bits)+1)
        waveform= Waveform()
        waveform.samples = samples
        waveform.sampling_frequency = sampling_freq
        bit_sequence[bit_sequence == 0] = -1
        waveform.data = np.complex128(bit_sequence[:samples]*np.cos(2.0*np.pi*carrier_freq * np.linspace(0,time,samples)))

        return waveform

    @staticmethod
    def awgn(time, sampling_frequency, center_freq, power, bw):
        waveform= Waveform()
        waveform.samples = int(time*sampling_frequency)
        waveform.sampling_frequency = sampling_frequency
        waveform.data = np.random.normal(0.0, 1, waveform.samples)
        waveform.bandpass_filter(center_freq - bw/2, center_freq + bw/2)
        waveform.data = 10**(power/10) / np.mean(abs(waveform.data)**2) * waveform.data
        return waveform

    @staticmethod
    def boc(samples, sampling_freq, subcarrier_freq, chipping_freq, bit_sequence=None):
        dt = 1/sampling_freq
        time = samples*dt
        if bit_sequence is None:
            bits = int((time)*chipping_freq)+1
            bit_sequence = np.random.randint(low=0, high=2, size=bits)
            bit_sequence = np.repeat(bit_sequence,int(samples/bits)+1)
        else:
            bit_sequence = copy.deepcopy(bit_sequence)
        waveform = Waveform()
        waveform.data = np.zeros(samples, dtype=np.complex128)
        waveform.sampling_frequency = sampling_freq
        waveform.samples = samples
        bit_sequence[bit_sequence == 0] = -1
        waveform.data = np.exp(1j*bit_sequence[:samples] * np.sign(np.sin(2.0*np.pi*subcarrier_freq * np.linspace(0,time,samples))))

        return waveform

def gps_ca(s1, s2):
    """
    https://www.gaussianwaves.com/2015/06/gold-code-generator/
    GPS C/A code (0/1, 1023 chips) for G2 phase-select stages s1, s2.
    """
    g1, g2, out = [1] * 10, [1] * 10, []
    for _ in range(1023):
        out.append(g1[9] ^ g2[s1 - 1] ^ g2[s2 - 1])
        f1 = g1[2] ^ g1[9]
        f2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1, g2 = [f1] + g1[:-1], [f2] + g2[:-1]
    return np.array(out)

class NavigationWaveform(Waveform):
    CODE_DEFINITION = {'CA': []}
    def __init__(self, data_bits, chip_duration):
        self.sampling_frequency = None
        self.samples = None
        self.data = None
        self.data_bits = data_bits
        self.chip_duration = chip_duration
        self.rel_time = np.linspace(0, len(data_bits)*chip_duration, len(data_bits))

    def get_bit(self, time_into_sequence):
        return self.data_bits[np.argmax(self.rel_time >= time_into_sequence)]

    @staticmethod
    def bpsk(samples, sampling_freq, carrier_freq, chipping_freq, code_type, prn, ):


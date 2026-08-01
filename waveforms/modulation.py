import numpy as np

class Waveform(object):
    def __init__(self):
        self.sampling_frequency = None
        self.data = None

    @staticmethod
    def sin(frequencies, amplitude, time, dt):
        waveform= Waveform()
        waveform.data = np.zeros(time.shape[0])
        waveform.sampling_frequency = 1.0/dt
        for freq, amp in zip(frequencies, amplitude):
            waveform.data += amp * np.sin(2.0*np.pi*freq * time)
        return waveform

    @staticmethod
    def bpsk(time, dt, carrier_freq, chipping_freq, bit_sequence=None):
        waveform= Waveform()
        waveform.data = np.zeros(time.shape[0])
        waveform.sampling_frequency = 1.0/dt

        return waveform

    @staticmethod
    def boc(time, dt, carrier_freq, chipping_freq, bit_sequence=None):
        if bit_sequence is None:
            bits = int((time[-1] - time[0])*chipping_freq)
            bit_sequence = np.random.randint(low=0, high=2, size=bits)
            bit_sequence = np.repeat(bit_sequence,int(time.shape[0]/bits)+1)
        waveform = Waveform()
        waveform.data = np.zeros(time.shape[0])
        waveform.sampling_frequency = 1.0 / dt
        bit_sequence[bit_sequence == 0] = -1
        waveform.data = np.exp(1j*bit_sequence[:time.shape[0]] * np.sign(np.sin(2.0*np.pi*carrier_freq * time[:])))

        return waveform

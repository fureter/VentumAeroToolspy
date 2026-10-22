import abc

import numpy as np

from utilities import coordinate_systems

class AcousticPattern(object):
    def __init__(self, gain, freq, az, el):
        self.gain = gain
        self.freq = freq
        self.az = az
        self.el = el

    @staticmethod
    def generate_ideal_array():
        pass

class Emitter(object):
    def __init__(self, init_position, init_attitude, emit_power, frequency, antenna_pattern):
        self.position = init_position
        self.attitude = init_attitude
        self.emit_power = emit_power
        self.base_frequency = frequency
        self.antenna_pattern = antenna_pattern

class Microphone(object):
    def __init__(self, init_position, init_attitude, antenna_pattern):
        self.position = init_position
        self.attitude = init_attitude
        self.antenna_pattern = antenna_pattern

class ElementArray(abc.ABC):
    def __init__(self, elements):
        self.elements = elements
        self.num_elem = len(elements)

        self.array_position = np.zeros(3)
        self.array_velocity = np.zeros(3)
        self.array_acceleration = np.zeros(3)
        self.array_attitude = np.zeros(3)

    def array_phase_offsets(self, az, el, frequency_range, speed_of_sound, phase_offsets=None):
        if phase_offsets is None:
            phase_offsets = np.zeros(self.num_elem)

        phase_delays = np.zeros([frequency_range.shape[0], self.num_elem])
        for elem in range(self.num_elem):
            wave_len = speed_of_sound / frequency_range
            phase_delays[:, elem] = phase_offsets[elem] + 2*np.pi*self.elements[elem].position @ np.array([[np.sin(np.pi/2 - el)*np.cos(az)],
                                                                                                   [np.sin(np.pi/2 - el)*np.sin(az)],
                                                                                                   [np.cos(el)]])/wave_len

        return phase_delays

    def apply_phase_delays(self, frequency_range, phase_delays, waveform):
        adjusted_waveform = np.zeros([frequency_range.shape[0], self.num_elem, waveform.data.shape[0]])

        wfm_fft, fft_freq = waveform.fft


        return adjusted_waveform


class EmitterArray(ElementArray):
    def __init__(self, emitters):
        super().__init__(emitters)

class MicrophoneArray(ElementArray):
    def __init__(self, microphones):
        super().__init__(microphones)

    def spatial_covariance(self, emitters, frequency_range, speed_of_sound):
        covar = np.zeros([self.num_elem, self.num_elem])
        for emitter in emitters:
            if isinstance(emitter, Emitter):
                r = emitter.position - self.array_position
                r_hat = r / np.linalg.norm(r, ord=2)
                r_hat_array_body = coordinate_systems.frd_dcm(self.array_position) @ r_hat
                az, el = coordinate_systems.get_frd_az_el(r_hat_array_body)
                phase_delays = self.array_phase_offsets(az, el, frequency_range, speed_of_sound)


            elif isinstance(emitter, EmitterArray):
                pass

        return covar

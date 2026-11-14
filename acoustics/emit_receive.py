import abc

import numpy as np

from antennas.antenna_pattern import AntennaPattern
from utilities import coordinate_systems

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
    def __init__(self, elements=None, array_pattern=None):
        self.elements = elements

        if elements is not None and array_pattern is None:
            self.array_pattern = AntennaPattern.from_elements(self.elements)
        elif elements is None and array_pattern is None:
            raise ValueError('Must specify at least one: elements or antenna_pattern')

        self.array_pattern = array_pattern
        self.num_elem = len(elements) if elements is not None else array_pattern.num_elem

        self.array_position = np.zeros(3)
        self.array_velocity = np.zeros(3)
        self.array_acceleration = np.zeros(3)
        self.array_attitude = np.zeros(3)
        self.last_excitation = None

    def get_array_response(self, az, el, frequency_range):
        return self.array_pattern.get_antenna_response(az, el, frequency_range)

    def get_array_response_with_excitation(self, az, el, frequency_range):
        if self.last_excitation is None:
            self.last_excitation = np.zeros(self.num_elem, dtype=np.complex128)
        return self.last_excitation.conj().T @ self.array_pattern.get_antenna_response(az, el, frequency_range)


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
                response = self.get_array_response(az, el, frequency_range)


            elif isinstance(emitter, EmitterArray):
                r = emitter.array_position - self.array_position
                r_hat = r / np.linalg.norm(r, ord=2)
                r_hat_array_body = coordinate_systems.frd_dcm(self.array_attitude) @ r_hat
                az, el = coordinate_systems.get_frd_az_el(r_hat_array_body)
                response = self.get_array_response(az, el, frequency_range)
                r_hat_emitter_body = coordinate_systems.frd_dcm(emitter.array_attitude) @ -r_hat
                emitter_az, emitter_el = coordinate_systems.get_frd_az_el(r_hat_emitter_body)
                emitter_response = emitter.get_array_response_with_excitation(emitter_az, emitter_el, frequency_range)



        return covar

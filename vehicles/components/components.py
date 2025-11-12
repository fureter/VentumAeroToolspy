import abc

import numpy as np

from utilities.coordinate_systems import rotation_matrix

class Component(object):
    def __init__(self, mass, position, angle, inertia_0):
        self.mass = mass
        self.cost = 0.0
        self.position = position
        self.angle = angle
        self.angle_rate = np.zeros(3)
        self.inertia_0 = inertia_0

        # Type is inhertited to upper level components and handled there
        self.type = None

    @property
    def inertia(self):
        return rotate_inertia(translate_inertia(self.inertia_0, self.position). self.angle)

    @property
    def angular_momentum(self):
        return rotation_matrix(self.angle) * (self.inertia_0 * self.angle_rate) * rotation_matrix(self.angle).T

    def log(self,logger):
        logger.info('Component: %s\r\tWeight: %s\r\tCost: %s\r\n' % (self.type, self.mass, self.cost))

    @abc.abstractmethod
    def calculate_loads_body_frame(self,vehicle, control_manager, environment_manager, simulation_manager):
        raise NotImplementedError('Not implemented for component type')

    @abc.abstractmethod
    def calculate_angular_momentum_body_frame(self, body_rate):
        raise NotImplementedError('Not implemented for component type')


@staticmethod
def rotate_inertia(inertia, angle):
    rot_mat = rotation_matrix(angle)
    return rot_mat * inertia * rot_mat.T


@staticmethod
def translate_inertia(inertia, mass, position):
    """

    :param np.matrix inertia:
    :param float mass:
    :param np.ndarray position:
    :return:
    """
    return inertia + mass * (position.dot(position)*np.eye(3) + np.outer(position, position))
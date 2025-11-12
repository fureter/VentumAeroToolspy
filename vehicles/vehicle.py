import abc
import logging

import numpy as np

from utilities.coordinate_systems import rotation_matrix


class Vehicle(object):

    def __init__(self, name, logger=None):
        self.components = list()
        self.mass = 0.0
        self.center_of_mass = np.zeros(3)
        self.inertia = np.identity(3)
        self.inv_inertia = np.linalg.inv(self.inertia)

        self.inertia_changed = False
        self.mass_changed = False

        self.states = np.zeros(12) # Translational and angular positions and rates

        self.name = name
        if logger is None:
            logger = logging.getLogger()
        self._logger = logger

    def add_component(self, component):
        self.components.append(component)
        self.inertia_changed = True
        self.mass_changed = True

    def update_inertia_and_mass(self):
        self.mass = 0.0
        self.inertia = np.matrix(np.zeros([3,3]))
        self.center_of_mass = np.zeros(3)

        for component in self.components:
            self.mass += component.mass
            self.inertia += component.inertia
            self.center_of_mass += component.mass*component.position

        self.center_of_mass /= self.mass
        self.inv_inertia = np.linalg.inv(self.inertia)


    @abc.abstractmethod
    def calculate_loads_body_frame(self, control_manager, environment_manager, simulation_manager):
        raise NotImplementedError('Not implemented for vehicle type')

    @abc.abstractmethod
    def calculate_angular_momentum_body_frame(self):
        raise NotImplementedError('Not implemented for vehicle type')

    @property
    def local_level_transform(self):
        return rotation_matrix(self.states[6:9])

    @property
    def velocity(self):
        return np.matrix([[self.state[3]],[self.state[4]],[self.state[5]]])

    @property
    def ang_rate(self):
        return np.matrix([[self.state[9]],[self.state[10]],[self.state[11]]])

    @property
    def omega_be_b(self):
        return np.matrix([[0, -self.state[8], self.state[7]],
                          [self.state[8], 0, -self.state[6]],
                          [-self.state[7], self.state[6], 0]])


    def log(self):
        for component in self.components:
            component.log(self._logger)



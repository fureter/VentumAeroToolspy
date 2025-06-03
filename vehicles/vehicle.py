import abc
import logging

import numpy as np

class Vehicle(object):

    def __init__(self, name, logger):
        self.components = list()
        self.mass = None
        self.inertia = np.identity(3)

        self.states = np.zeros(12) # Translational and angular positions and rates

        self.name = name
        if logger is None:
            logger = logging.getLogger()
        self._logger = logger

    @abc.abstractmethod
    def update_forces(self, control_manager, environment_manager):
        raise NotImplementedError('Not implemented for vehicle type')

    def add_component(self, component):
        self.components.append(component)

    def log(self):
        for component in self.components:
            component.log(self._logger)



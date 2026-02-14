import abc
import logging

import numpy as np
from scipy.spatial.transform import Rotation

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

        self.states = np.zeros(13) # Translational and angular positions and rates
        self.previous_states = np.zeros(13)
        self.force = np.zeros(3)
        self.moment = np.zeros(3)

        self.num_components = 0

        self.name = name
        if logger is None:
            logger = logging.getLogger()
        self._logger = logger

    def add_component(self, component):
        self.components.append(component)
        self.components[-1].num_id = self.num_components
        self.num_components += 1
        self.inertia_changed = True
        self.mass_changed = True

    def update_inertia_and_mass(self):
        self.mass = 0.0
        self.inertia = np.array(np.zeros([3,3]))
        self.center_of_mass = np.zeros(3)

        for component in self.components:
            self.mass += component.mass
            self.inertia += component.inertia
            self.center_of_mass += component.mass*component.position

        self.center_of_mass /= self.mass
        self.inv_inertia = np.linalg.inv(self.inertia)

    def move_origin_to_center_of_mass(self):
        for component in self.components:
            component.position = component.position - self.center_of_mass
        self.update_inertia_and_mass()

    @abc.abstractmethod
    def calculate_loads_body_frame(self, control_manager, environment_manager, simulation_manager):
        raise NotImplementedError('Not implemented for vehicle type')

    @abc.abstractmethod
    def calculate_angular_momentum_body_frame(self):
        raise NotImplementedError('Not implemented for vehicle type')

    @abc.abstractmethod
    def control_input(self, control_manager):
        raise NotImplementedError('Not implemented for vehicle type')

    @property
    def local_level_transform(self):
        return self.body_transform.T

    @property
    def body_transform(self):
        return np.array([[self.q0**2 + self.q1**2 - self.q2**2 - self.q3**2,    2*(self.q1*self.q2+self.q0*self.q3),                    2*(self.q1*self.q3 - self.q0*self.q2)],
                         [2*(self.q1*self.q2-self.q0*self.q3),                  self.q0**2 - self.q1**2 + self.q2**2 - self.q3**2,      2*(self.q2*self.q3+self.q0*self.q1)],
                         [2*(self.q1*self.q3+self.q0*self.q2),                  2*(self.q2*self.q3-self.q0*self.q1),                    self.q0**2 - self.q1**2 - self.q2**2 + self.q3**2]])

    @property
    def euler_to_body_rate_transform(self):
        return np.array([[1.0, 0, -np.sin(self.attitude[1])],
                         [0.0, np.cos(self.attitude[0]), np.sin(self.attitude[0]) * np.cos(self.attitude[1])],
                         [0.0, -np.sin(self.attitude[0]), np.cos(self.attitude[0])*np.cos(self.attitude[1])]])

    @property
    def position(self):
        return np.array([self.states[0],self.states[1],self.states[2]])

    @property
    def velocity(self):
        """
        North East Down Coordinate System
        :return:
        """
        return np.array([self.states[3],self.states[4],self.states[5]])

    @property
    def body_velocity(self):
        return self.body_transform @ np.array([self.states[3],self.states[4],self.states[5]])

    @property
    def roll(self):
        quat = self.states[6:10]
        return np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]), 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))

    @property
    def pitch(self):
        quat = self.states[6:10]
        return np.arcsin(2 * (quat[0] * quat[2] - quat[3] * quat[1]))

    @property
    def yaw(self):
        quat = self.states[6:10]
        return np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))

    @property
    def attitude(self):
        quat = self.states[6:10]
        pitch = np.arcsin(2*(quat[0]*quat[2] - quat[3]*quat[1]))
        roll = np.arctan2(2*(quat[0]*quat[1] + quat[2]*quat[3]), 1.0 - 2.0*(quat[1]**2 + quat[2]**2))
        yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1.0 - 2.0*(quat[2]**2 + quat[3]**2))
        return np.array([roll, pitch, yaw])

    @property
    def ang_rate(self):
        return np.array([self.states[10],self.states[11],self.states[12]])

    @property
    def p(self):
        return self.states[10]

    @property
    def q(self):
        return self.states[11]

    @property
    def r(self):
        return self.states[12]

    @property
    def q0(self):
        return self.states[6]

    @property
    def q1(self):
        return self.states[7]

    @property
    def q2(self):
        return self.states[8]

    @property
    def q3(self):
        return self.states[9]

    @property
    def omega_be_b(self):
        return np.array([[0, -self.r, self.q],
                          [self.r, 0, -self.p],
                          [-self.q, self.p, 0]])

    @property
    def att_rate_skew_matrix(self):
        return np.array([[0, -self.p, -self.q, -self.r],
                         [self.p, 0, self.r, -self.q],
                         [self.q, -self.r, 0, self.p],
                         [self.r, self.q, -self.p, 0]])


    def log(self):
        for component in self.components:
            component.log(self._logger)



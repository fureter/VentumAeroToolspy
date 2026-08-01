import abc
import logging

import numpy as np
from scipy.spatial.transform import Rotation

from utilities.coordinate_systems import frd_dcm

class DefaultVehicleLoggingKeys(object):
    MASS = 'mass'

    COM_X = 'center of mass x'
    COM_Y = 'center of mass y'
    COM_Z = 'center of mass z'
    COM_KEYS = [COM_X, COM_Y, COM_Z]

    INERTIA_XX = 'Intertia xx'
    INERTIA_YY = 'Intertia yy'
    INERTIA_ZZ = 'Intertia zz'
    INERTIA_XY = 'Intertia xy'
    INERTIA_XZ = 'Intertia xz'
    INERTIA_YZ = 'Intertia yz'
    INERTIA_KEYS = [INERTIA_XX, INERTIA_YY, INERTIA_ZZ, INERTIA_XY, INERTIA_XZ, INERTIA_YZ]

    ALL_KEYS = [MASS, COM_X, COM_Y, COM_Z, INERTIA_XX, INERTIA_YY, INERTIA_ZZ, INERTIA_XY, INERTIA_XZ, INERTIA_YZ]

class Vehicle(object):

    def __init__(self, name, data_logger):
        self.components = list()
        self.mass = 0.0
        self.center_of_mass = np.zeros(3)
        self.inertia = np.identity(3)
        self.inv_inertia = np.linalg.inv(self.inertia)

        self.inertia_changed = False
        self.mass_changed = False

        self.data_logger = data_logger
        self.data_logger.register_items(DefaultVehicleLoggingKeys.ALL_KEYS)

        self.states = np.zeros(13) # Translational and angular positions and rates
        self.previous_states = np.zeros(13)
        self.force = np.zeros(3)
        self.moment = np.zeros(3)

        self.num_components = 0

        self.name = name

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
    def update_vehicle(self, control_manager, environment_manager, simulation_manager):
        raise NotImplementedError('Not implemented for vehicle type')

    def calculate_loads_body_frame(self, control_manager, environment_manager, simulation_manager):
        force = np.zeros(3)
        moment = np.zeros(3)

        for component in self.components:
            f, m = component.calculate_loads_body_frame(self, control_manager, environment_manager, simulation_manager)
            force += f
            moment += m
        self.force = force
        self.moment = moment

        return force, moment

    def calculate_angular_momentum_body_frame(self):
        angular_momentum = np.array(np.zeros(3))
        for component in self.components:
            angular_momentum += component.calculate_angular_momentum_body_frame(self.ang_rate)

        return angular_momentum

    @abc.abstractmethod
    def control_input(self, control_manager):
        raise NotImplementedError('Not implemented for vehicle type')


    def log_iteration(self, time):
        self.data_logger.add_data(time, item=DefaultVehicleLoggingKeys.MASS, data=self.mass)
        self.data_logger.add_data_items(time, items=DefaultVehicleLoggingKeys.COM_KEYS, datas=self.center_of_mass)
        self.data_logger.add_data(time, item=DefaultVehicleLoggingKeys.INERTIA_XX, data=self.inertia[0,0])
        self.data_logger.add_data(time, item=DefaultVehicleLoggingKeys.INERTIA_YY, data=self.inertia[1,1])
        self.data_logger.add_data(time, item=DefaultVehicleLoggingKeys.INERTIA_ZZ, data=self.inertia[2,2])
        self.data_logger.add_data(time, item=DefaultVehicleLoggingKeys.INERTIA_XY, data=self.inertia[0,1])
        self.data_logger.add_data(time, item=DefaultVehicleLoggingKeys.INERTIA_XZ, data=self.inertia[0,2])
        self.data_logger.add_data(time, item=DefaultVehicleLoggingKeys.INERTIA_YZ, data=self.inertia[1,2])

    def plot_default_parameters(self):
        self.data_logger.simple_plot(DefaultVehicleLoggingKeys.MASS, time_frame=None,
                                     subplot=False)

        self.data_logger.simple_plot(DefaultVehicleLoggingKeys.COM_KEYS, time_frame=None,
                                     subplot=False)

        self.data_logger.simple_plot(DefaultVehicleLoggingKeys.INERTIA_KEYS, time_frame=None,
                                     subplot=False)

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
    def body_rate_to_euler_transform(self):
        att = self.attitude
        return np.array([[1.0, np.sin(att[0])*np.tan(att[1]), np.cos(att[0])*np.tan(att[1])],
                         [0.0, np.cos(att[1]), -np.sin(att[0])],
                         [0.0, np.sin(att[0])/np.cos(att[1]), np.cos(att[1])/np.cos(att[1])]])

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

    @property
    def inertia_tensor(self):
        inertia = self.inertia
        return np.array([[inertia[0,0], -inertia[0,1], -inertia[0,2]],
                         [-inertia[1,0], inertia[1,1], -inertia[1,2]],
                         [-inertia[2,0], -inertia[2,1], inertia[2,2]]])

    def log(self):
        for component in self.components:
            component.log(self._logger)



import abc
from abc import ABC

import numpy as np

from vehicles.components.components import Component

class Propulsion(Component, ABC):
    def __init__(self, mass, position, angle, inertia_0, num_id=0):
        super().__init__(mass, position, angle, inertia_0, num_id)

    @abc.abstractmethod
    def get_thrust(self,vehicle, control_manager, environment_manager, simulation_manager):
        raise NotImplementedError('Not implemented')

    @abc.abstractmethod
    def process_fuel_consumption(self,vehicle, control_manager, environment_manager, simulation_manager):
        raise NotImplementedError('Not implemented')

class IdealSolidRocket(Propulsion):
    def __init__(self, mass, position, angle, specific_isp, average_thrust, fuel_mass_fraction, num_id=0):


        length = 0.2
        outer_radius = 0.1
        inertia_0 = np.array([[0.5 * mass * (outer_radius ** 2), 0.0, 0.0],
                              [0.0, 1.0 / 12.0 * mass * (3 * (outer_radius ** 2) + length ** 2),
                               0.0],
                              [0.0, 0.0,
                               1.0 / 12.0 * mass * (3 * (outer_radius ** 2) + length ** 2)]])
        super().__init__(mass, position, angle, inertia_0, num_id)

        self.isp = specific_isp
        self.average_thrust = average_thrust
        self.fuel_mass = fuel_mass_fraction * mass
        self.burn_time = self.isp/self.average_thrust*self.fuel_mass
        self.empty_mass = mass - self.fuel_mass

        self.burn_complete = False
        self.burn_start = False

    def calculate_loads_body_frame(self, vehicle, control_manager, environment_manager, simulation_manager):
        thrust = np.array([self.get_thrust(vehicle, control_manager, environment_manager, simulation_manager), 0.0, 0.0])
        force = self.body_to_component_transform.T @ thrust
        moment = np.cross(self.position - vehicle.center_of_mass, force)

        return force, moment


    def geom3d(self):
        return None, None, None


    def calculate_angular_momentum_body_frame(self, body_rate):
        return np.zeros(3)

    def get_thrust(self, vehicle, control_manager, environment_manager, simulation_manager):
        if self.burn_start and not self.burn_complete:
            return self.average_thrust
        return 0.0

    def process_fuel_consumption(self, vehicle, control_manager, environment_manager, simulation_manager):
        if self.burn_start and not self.burn_complete:
            vehicle.mass_changed = True
            fuel_consumption = self.average_thrust*simulation_manager.dt/self.isp
            if self.fuel_mass >= fuel_consumption:
                self.fuel_mass -= fuel_consumption
                self.mass = self.fuel_mass + self.empty_mass
            else:
                self.burn_complete = True
                self.fuel_mass = 0.0
                self.mass = self.empty_mass
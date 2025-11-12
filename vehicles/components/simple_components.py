import numpy as np

from utilities.coordinate_systems import rotation_matrix
from .components import  Component

class ShellFuselage(Component):
    def __init__(self, mass, outer_radius, inner_radius, length, position, angle):
        inertia_0 = np.matrix([[1.0/12.0 * mass * (3* (outer_radius**2 + inner_radius**2) + length**2), 0.0, 0.0],
                               [0.0, 1.0/12.0 * mass * (3* (outer_radius**2 + inner_radius**2) + length**2), 0.0],
                               [0.0, 0.0, 0.5 * mass * (outer_radius**2 + inner_radius**2)]])
        super.__init__(mass, position, angle, inertia_0)

    def calculate_loads_body_frame(self, vehicle, control_manager, environment_manager, simulation_manager):
        raise NotImplementedError('Not implemented for component type')

    def calculate_angular_momentum_body_frame(self, body_rate):
        return (self.inertia * body_rate + rotation_matrix(self.angle) * self.inertia_0 *
                self.angular_momentum * rotation_matrix(self.angle).T)
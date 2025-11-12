import numpy as np

from .vehicle import Vehicle

class SimpleArtillery(Vehicle):
    def __init__(self, name, logger=None):
        super.__init__(name=name, logger=logger)

    def calculate_loads_body_frame(self, control_manager, environment_manager, simulation_manager):
        force = np.zeros(3)
        moment = np.zeros(3)

        for component in self.components:
            f, m = component.calculate_force(self, control_manager, environment_manager, simulation_manager)
            force += f
            moment += m + (component.position - self.center_of_mass) * f

        return force, moment

    def calculate_angular_momentum_body_frame(self):
        angular_momentum = np.matrix(np.zeros(3))
        for component in self.components:
            angular_momentum += component.calculate_angular_momentum_body_frame(self.states[9:12])

        return angular_momentum

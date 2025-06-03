class SimulationManager(object):
    def __init__(self, pos_init, vel_init, angle_init, angle_rate_init):
        self.initial_pos = pos_init
        self.initial_vel = vel_init
        self.initial_attitude = angle_init
        self.initial_attitude_rate = angle_rate_init

class EnvironmentManager(object):
    def __init__(self):
        pass

    def air_properties(self, altitude):
        pass
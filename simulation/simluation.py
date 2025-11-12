class SimulationManager(object):
    def __init__(self, pos_init, vel_init, angle_init, angle_rate_init, log):
        self.initial_pos = pos_init
        self.initial_vel = vel_init
        self.initial_attitude = angle_init
        self.initial_attitude_rate = angle_rate_init
        self.logger = log
        self.simulate = True

    def update(self):
        pass

class EnvironmentManager(object):
    def __init__(self):
        pass

    def update(self):
        pass

    def air_properties(self, altitude):
        pass

class ControlManager():
    def __init__(self):
        pass

    def update(self):
        pass

class DataLogger(object):
    def __init__(self):
        self.data = dict()

import numpy as np


class EnvironmentManager(object):
    def __init__(self, data_logger):
        # self.swfile = download_sw_nrlmsise00()
        # self.swdata = read_sw_nrlmsise00(self.swfile)
        # self.t = '2014-07-22 22:18:45'
        # self.nrl00 = nrlmsise00(self.t,(0,0,0),self.swdata)

        self.wind_velocity = np.zeros(3)

        self.data_logger = data_logger

    def update(self, vehicle, simulation_manager):
        # self.nrl00 = nrlmsise00(simulation_manager.utc_time,
        #                    (np.rad2deg(simulation_manager.lla[0]), np.rad2deg(simulation_manager.lla[1]), simulation_manager.lla[2]/1000.0),
        #                    self.swdata)
        pass

    def air_properties(self):
        return 1.225, 25.0  # self.nrl00.rho, self.nrl00.T

    def gravity(self, altitude):
        return np.array([0, 0, 9.81])

    def log_iteration(self, time):
        pass

    # def omega_be_b(self):
    #     return np.array([[0, -self.states[11], self.states[10]],
    #                       [self.states[11], 0, -self.states[9]],
    #                       [-self.states[10], self.states[9], 0]])

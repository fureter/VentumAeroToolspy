from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
# from pyatmos import nrlmsise00, download_sw_nrlmsise00, read_sw_nrlmsise00

from utilities import coordinate_systems


class SimulationManager(object):
    class DefaultSimulationLoggingKeys(object):
        NORTH_POSITION = 'north_position'
        EAST_POSITION = 'east_position'
        DOWN_POSITION = 'down_position'
        POSITION_KEYS = [NORTH_POSITION, EAST_POSITION, DOWN_POSITION]

        NORTH_VELOCITY = 'north_velocity'
        EAST_VELOCITY = 'east_velocity'
        DOWN_VELOCITY = 'down_velocity'
        NED_VELOCITY_MAG = 'ned_velocity_mag'
        VELOCITY_KEYS = [NORTH_VELOCITY, EAST_VELOCITY, DOWN_VELOCITY]
        ALL_VELOCITY_KEYS = [NORTH_VELOCITY, EAST_VELOCITY, DOWN_VELOCITY] + [NED_VELOCITY_MAG]

        ROLL = 'roll'
        PITCH = 'pitch'
        YAW = 'yaw'
        ATTITUDE_KEYS = [ROLL, PITCH, YAW]

        ROLL_RATE = 'roll_rate'
        PITCH_RATE = 'pitch_rate'
        YAW_RATE = 'yaw_rate'
        ATTITUDE_RATE_KEYS = [ROLL_RATE, PITCH_RATE, YAW_RATE]

        FORCE_X = 'force_x'
        FORCE_Y = 'force_y'
        FORCE_Z = 'force_z'
        FORCE_KEYS = [FORCE_X, FORCE_Y, FORCE_Z]

        MOMENT_X = 'moment_x'
        MOMENT_Y = 'moment_y'
        MOMENT_Z = 'moment_z'
        MOMENT_KEYS = [MOMENT_X, MOMENT_Y, MOMENT_Z]

        ALL_KEYS = POSITION_KEYS + ALL_VELOCITY_KEYS + ATTITUDE_KEYS + ATTITUDE_RATE_KEYS + FORCE_KEYS + MOMENT_KEYS

    def __init__(self, pos_init, vel_init, angle_init, angle_rate_init, log, sim_rate, max_runtime):
        self.initial_pos = pos_init
        self.initial_vel = vel_init
        self.initial_attitude = angle_init
        self.initial_attitude_rate = angle_rate_init

        self.data_logger = log

        self.data_logger.register_items(items=SimulationManager.DefaultSimulationLoggingKeys.ALL_KEYS)

        self.pos_hist = [pos_init]
        self.vel_hist = [vel_init]
        self.att_hist = [angle_init]
        self.att_rate_hist = [angle_rate_init]
        self.force_hist = [[0,0,0]]
        self.moment_hist = [[0,0,0]]

        self.rate = sim_rate
        self.dt = 1.0 / sim_rate
        self.time = 0
        self.max_runtime = max_runtime
        self.logger = log
        self.simulate = True

        self.lla = np.array([0,0,pos_init[-1]])
        self.utc_time = '2014-07-22 22:18:45'

    def lla_init(self, lla):
        self.lla = lla

    def lla_update(self, vehicle):
        ecef = coordinate_systems.lla_to_ecef(self.lla)
        dned = vehicle.states[0:3] - vehicle.previous_states[0:3]
        d_ecef = coordinate_systems.ned_to_ecef(self.lla, dned)
        new_ecef = ecef + d_ecef
        self.lla = coordinate_systems.ecef_to_lla(new_ecef)

    def update(self, vehicle, control_manager, environment_manager, log_data=True):
        self.lla_update(vehicle)
        self.time += self.dt
        if self.time > self.max_runtime or self.lla[2] < -100:
            self.simulate = False
        if log_data:
            self.log_iteration(vehicle, control_manager, environment_manager)

    def log_iteration(self, vehicle, control_manager, environment_manager):
        self.data_logger.add_data_items(self.time, items=SimulationManager.DefaultSimulationLoggingKeys.POSITION_KEYS, datas=vehicle.position)
        self.data_logger.add_data_items(self.time, items=SimulationManager.DefaultSimulationLoggingKeys.VELOCITY_KEYS, datas=vehicle.velocity)
        self.data_logger.add_data(self.time, item=SimulationManager.DefaultSimulationLoggingKeys.NED_VELOCITY_MAG, data=np.linalg.norm(vehicle.velocity, ord=2))
        self.data_logger.add_data_items(self.time, items=SimulationManager.DefaultSimulationLoggingKeys.ATTITUDE_KEYS, datas=np.rad2deg(vehicle.attitude))
        self.data_logger.add_data_items(self.time, items=SimulationManager.DefaultSimulationLoggingKeys.ATTITUDE_RATE_KEYS, datas=np.rad2deg(vehicle.ang_rate))
        self.data_logger.add_data_items(self.time, items=SimulationManager.DefaultSimulationLoggingKeys.FORCE_KEYS, datas=vehicle.force)
        self.data_logger.add_data_items(self.time, items=SimulationManager.DefaultSimulationLoggingKeys.MOMENT_KEYS, datas=vehicle.moment)

        vehicle.log_iteration(self.time)
        control_manager.log_iteration(self.time)
        environment_manager.log_iteration(self.time)

    def generate_all_plots(self):
        self.plot_position()
        self.plot_velocity()
        self.plot_angle()
        self.plot_angle_rate()
        self.plot_force()
        self.plot_moment()

    def plot_position(self):
        self.data_logger.simple_plot(SimulationManager.DefaultSimulationLoggingKeys.POSITION_KEYS, time_frame=None, subplot=False)

    def plot_angle(self):
        self.data_logger.simple_plot(SimulationManager.DefaultSimulationLoggingKeys.ATTITUDE_KEYS, time_frame=None, subplot=False)

    def plot_angle_rate(self):
        self.data_logger.simple_plot(SimulationManager.DefaultSimulationLoggingKeys.ATTITUDE_RATE_KEYS, time_frame=None, subplot=False)

    def plot_velocity(self):
        self.data_logger.simple_plot(SimulationManager.DefaultSimulationLoggingKeys.ALL_VELOCITY_KEYS, time_frame=None, subplot=False)

    def plot_force(self):
        self.data_logger.simple_plot(SimulationManager.DefaultSimulationLoggingKeys.FORCE_KEYS, time_frame=None, subplot=False)

    def plot_moment(self):
        self.data_logger.simple_plot(SimulationManager.DefaultSimulationLoggingKeys.MOMENT_KEYS, time_frame=None, subplot=False)


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
        return  1.225, 25.0#self.nrl00.rho, self.nrl00.T
    
    def gravity(self, altitude):
        return np.array([0,0,9.81])

    def log_iteration(self, time):
        pass

    # def omega_be_b(self):
    #     return np.array([[0, -self.states[11], self.states[10]],
    #                       [self.states[11], 0, -self.states[9]],
    #                       [-self.states[10], self.states[9], 0]])

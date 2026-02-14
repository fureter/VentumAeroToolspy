import numpy as np
from matplotlib import pyplot as plt
from pyatmos import nrlmsise00, download_sw_nrlmsise00, read_sw_nrlmsise00

from utilities import coordinate_systems


class SimulationManager(object):
    def __init__(self, pos_init, vel_init, angle_init, angle_rate_init, log, sim_rate, max_runtime):
        self.initial_pos = pos_init
        self.initial_vel = vel_init
        self.initial_attitude = angle_init
        self.initial_attitude_rate = angle_rate_init

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

    def update(self, vehicle, control_manager, environment_manager):
        self.lla_update(vehicle)
        self.time += self.dt
        if self.time > self.max_runtime or self.lla[2] < -100:
            self.simulate = False
        self.log_iteration(vehicle, control_manager, environment_manager)

    def log_iteration(self, vehicle, control_manager, environment_manager):
        self.pos_hist.append(vehicle.position)
        self.vel_hist.append(vehicle.velocity)
        self.att_hist.append(vehicle.attitude)
        self.att_rate_hist.append(vehicle.ang_rate)
        self.force_hist.append(vehicle.force)
        self.moment_hist.append(vehicle.moment)

    def plot_position(self):
        plt.figure()
        pos = np.array(self.pos_hist)
        time = np.linspace(0, self.time+self.dt, pos.shape[0], endpoint=False)
        plt.plot(time, pos[:,0])
        plt.plot(time, pos[:,1])
        plt.plot(time, pos[:,2])
        plt.legend(['North', 'East', 'Down'])
        plt.grid()
        # plt.show()

    def plot_angle(self):
        plt.figure()
        pos = np.array(self.att_hist)
        time = np.linspace(0, self.time + self.dt, pos.shape[0], endpoint=False)
        plt.plot(time, np.rad2deg(np.unwrap(pos[:, 0])))
        plt.plot(time, np.rad2deg(np.unwrap(pos[:, 1])))
        plt.plot(time, np.rad2deg(np.unwrap(pos[:, 2])))
        plt.legend(['roll', 'pitch', 'yaw'])
        plt.grid()
        # plt.show()

    def plot_angle_rate(self):
        plt.figure()
        pos = np.array(self.att_rate_hist)
        time = np.linspace(0, self.time + self.dt, pos.shape[0], endpoint=False)
        plt.plot(time, pos[:, 0])
        plt.plot(time, pos[:, 1])
        plt.plot(time, pos[:, 2])
        plt.legend(['roll rt', 'pitch rt', 'yaw rt'])
        plt.grid()
        # plt.show()

    def plot_velocity(self):
        plt.figure()
        vel = np.array(self.vel_hist)
        time = np.linspace(0, self.time+self.dt, vel.shape[0], endpoint=False)
        plt.plot(time, vel[:,0])
        plt.plot(time, vel[:,1])
        plt.plot(time, vel[:,2])
        plt.legend(['North Vel', 'East Vel', 'Down Vel'])
        plt.title('NED Velocity')
        plt.grid()
        # plt.show()

    def plot_force(self):
        plt.figure()
        vel = np.array(self.force_hist)
        time = np.linspace(0, self.time+self.dt, vel.shape[0], endpoint=False)
        plt.plot(time, vel[:,0])
        plt.plot(time, vel[:,1])
        plt.plot(time, vel[:,2])
        plt.legend(['Fx', 'Fy', 'Fz'])
        plt.grid()
        # plt.show()

    def plot_moment(self):
        plt.figure()
        vel = np.array(self.moment_hist)
        time = np.linspace(0, self.time+self.dt, vel.shape[0], endpoint=False)
        plt.plot(time, vel[:,0])
        plt.plot(time, vel[:,1])
        plt.plot(time, vel[:,2])
        plt.legend(['Mp', 'Mq', 'Mr'])
        plt.grid()
        # plt.show()


class EnvironmentManager(object):
    def __init__(self):
        self.swfile = download_sw_nrlmsise00()
        self.swdata = read_sw_nrlmsise00(self.swfile)
        self.t = '2014-07-22 22:18:45'
        self.nrl00 = nrlmsise00(self.t,(0,0,0),self.swdata)

        self.wind_velocity = np.zeros(3)

    def update(self, vehicle, simulation_manager):
        self.nrl00 = nrlmsise00(simulation_manager.utc_time,
                           (np.rad2deg(simulation_manager.lla[0]), np.rad2deg(simulation_manager.lla[1]), simulation_manager.lla[2]/1000.0),
                           self.swdata)

    def air_properties(self):
        return self.nrl00.rho, self.nrl00.T
    
    def gravity(self, altitude):
        return np.array([0,0,9.81])

    # def omega_be_b(self):
    #     return np.array([[0, -self.states[11], self.states[10]],
    #                       [self.states[11], 0, -self.states[9]],
    #                       [-self.states[10], self.states[9], 0]])


class DataLogger(object):
    def __init__(self):
        self.data = dict()

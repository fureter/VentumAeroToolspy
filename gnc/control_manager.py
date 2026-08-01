import numpy as np

from gnc.pid import PID


class ControlManager(object):
    def __init__(self, data_logger, update_rate=400):
        self.roll_out = 0
        self.pitch_out = 0
        self.yaw_out = 0
        self.update_rate = update_rate
        self.last_update = 0.0

        self.data_logger = data_logger
        self.active = False

    def update(self, vehicle, simulation_manager):
        pass

    def log_iteration(self, time):
        pass

class RateController(ControlManager):
    def __init__(self, data_logger, update_rate=400):
        super().__init__(data_logger, update_rate)
        self.roll_rate_target = 0.0
        self.pitch_rate_target = 0.0
        self.yaw_rate_target = 0.0

        self.roll_pid = PID(kp=0.5, ki=0.0, kd=0.2)
        self.yaw_pid = PID(kp=0.1, ki=0.0, kd=0.1)
        self.pitch_pid = PID(kp=0.1, ki=0.0, kd=0.1)

    def update(self, vehicle, simulation_manager):
        if simulation_manager.time > 1.0:
            self.active = True
        if self.active:
            self.roll_pid.update(vehicle.p, self.roll_rate_target, simulation_manager.dt)
            self.pitch_pid.update(vehicle.q, self.pitch_rate_target, simulation_manager.dt)
            self.yaw_pid.update(vehicle.r, self.yaw_rate_target, simulation_manager.dt)

            self.roll_out = self.roll_pid.output
            self.pitch_out = self.pitch_pid.output
            self.yaw_out = self.yaw_pid.output

class AttitudeController(ControlManager):
    def __init__(self, data_logger, update_rate=400):
        super().__init__(data_logger, update_rate)
        self.roll_target = 0.0
        self.pitch_target = np.deg2rad(10.0)
        self.yaw_target = 0.0

        self.roll_pid = PID(kp=2.5, ki=0.01, kd=0.5)
        self.yaw_pid = PID(kp=0.5, ki=0.05, kd=0.5)
        self.pitch_pid = PID(kp=2.5, ki=0.05, kd=0.2)

    def update(self, vehicle, simulation_manager):
        if self.active:
            if self.last_update + 1.0/self.update_rate <= simulation_manager.time:
                self.last_update = simulation_manager.time
                self.roll_pid.update(vehicle.roll, self.roll_target, max(simulation_manager.dt, 1.0/self.update_rate))
                self.pitch_pid.update(vehicle.pitch, self.pitch_target, max(simulation_manager.dt, 1.0/self.update_rate))
                self.yaw_pid.update(vehicle.yaw, self.yaw_target, max(simulation_manager.dt, 1.0/self.update_rate))

            self.roll_out = self.roll_pid.output
            self.pitch_out = self.pitch_pid.output
            self.yaw_out = self.yaw_pid.output
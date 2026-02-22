import numpy as np

from gnc.pid import PID


class ControlManager(object):
    def __init__(self):
        self.roll_out = 0
        self.pitch_out = 0
        self.yaw_out = 0

        self.active = False

    def update(self, vehicle, simulation_manager):
        pass

class RateController(ControlManager):
    def __init__(self):
        super().__init__()
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
    def __init__(self):
        super().__init__()
        self.roll_target = 0.0
        self.pitch_target = np.deg2rad(-1.0)
        self.yaw_target = 0.0

        self.roll_pid = PID(kp=2.5, ki=0.5, kd=1.5)
        self.yaw_pid = PID(kp=0.5, ki=0.5, kd=0.5)
        self.pitch_pid = PID(kp=0.5, ki=0.5, kd=0.2)

    def update(self, vehicle, simulation_manager):
        self.yaw_target = vehicle.yaw
        if vehicle.velocity[2] > 0.0:
            self.active = True
        if self.active:
            self.roll_pid.update(vehicle.roll, self.roll_target, simulation_manager.dt)
            self.pitch_pid.update(vehicle.pitch, self.pitch_target, simulation_manager.dt)
            self.yaw_pid.update(vehicle.yaw, self.yaw_target, simulation_manager.dt)

            self.roll_out = self.roll_pid.output
            self.pitch_out = self.pitch_pid.output
            self.yaw_out = self.yaw_pid.output
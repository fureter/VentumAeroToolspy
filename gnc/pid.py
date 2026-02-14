class PID(object):
    def __init__(self, kp, kd, ki):
        self.kp = kp
        self.kd = kd
        self.ki = ki

        self.target = 0
        self.error = 0
        self.previous_error = 0
        self.integral_error = 0
        self.output = 0

    def update(self, value, target, dt):
        self.target = target
        self.error = target - value

        de = (self.error + self.previous_error)/dt
        self.integral_error += dt * self.error

        self.output = self.kp * self.error + self.kd * de + self.ki * self.integral_error
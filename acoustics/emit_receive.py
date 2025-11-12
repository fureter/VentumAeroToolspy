
class AcousticPattern(object):
    def __init__(self):
        pass

class Emitter(object):
    def __init__(self, init_position, init_attitude, emit_power, frequency, antenna_pattern):
        self.position = init_position
        self.attitude = init_attitude
        self.emit_power = emit_power
        self.base_frequency = frequency
        self.antenna_pattern = antenna_pattern

class Microphone(object):
    def __init__(self, init_position, init_attitude, antenna_pattern):
        self.position = init_position
        self.attitude = init_attitude
        self.antenna_pattern = antenna_pattern

class CompositeMicrophoneArray(object):
    def __init__(self, microphones):
        self.microphones = microphones

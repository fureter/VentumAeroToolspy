import abc

class Sensor(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def observable(self, vehicle, environment_manager, simulation_manager):
        raise NotImplementedError
import logging

from .Components import Components

class Vehicle(object):

    def __init__(self, name, logger):
        self.components = list()

        self.name = name
        if logger is None:
            logger = logging.getLogger()
        self._logger = logger

    def log(self):
        for component in self.components:
            component.log(self._logger)



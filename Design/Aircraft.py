import logging
from matplotlib import pyplot as plt

from .Vehicle import Vehicle
from .Components.Wing import Wing


class Aircraft(Vehicle):

    def __init__(self, name, logger=None):

        super().__init__(name, logger)

    def add_component(self, component, position):
        self.components.append((component, position))


    def plot_planform(self):
        legend = list()
        for com in self.components:
            (x,y) = com[0].planform_coordinates()
            x += com[1][1]
            y += com[1][0]
            plt.plot(x,y)
            legend.append('%s:%s' % (com[0].type,com[0].sub_type))

        plt.xlabel('y (m)')
        plt.ylabel('x (m)')
        plt.legend(legend)
        plt.title('Aircraft: %s' % self.name)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
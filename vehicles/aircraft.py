from matplotlib import pyplot as plt

from .vehicle import Vehicle


class Aircraft(Vehicle):

    def __init__(self, name, logger=None):

        super().__init__(name, logger)


    def update_forces(self, control_manager, environment_manager):
        raise NotImplementedError('Not implemented for vehicle type')

    def plot_planform(self, half=False):
        legend = list()
        for com in self.components:
            (x,y) = com[0].planform_coordinates(half)
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
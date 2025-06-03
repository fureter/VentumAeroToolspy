import numpy as np
import scipy

from vehicles.vehicle import Vehicle
from simulation.simluation import SimulationManager

def simulate_6dof_equations(t, y, vehicle, control_manager, environment_manager):


def simulate_vehicle(vehicle, control_manager, environment_manager, simulation_manager):
    """

    :param Vehicle vehicle:
    :param control_manager:
    :param environment_manager:
    :param SimulationManager simulation_manager:
    :return:
    """

    # initialize vehicle states
    vehicle.states[0:3] = simulation_manager.initial_pos
    vehicle.states[3:6] = simulation_manager.initial_vel
    vehicle.states[6:9] = simulation_manager.initial_attitude
    vehicle.states[9:] = simulation_manager.initial_attitude_rate




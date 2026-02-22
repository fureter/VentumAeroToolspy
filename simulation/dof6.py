import copy
from threading import Thread

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from utilities.coordinate_systems import euler_to_quaternion
from vehicles import vehicle
from vehicles.vehicle import Vehicle
from simulation.simluation import SimulationManager

def initialize_simulation(vehicles, control_manager, environment_manager, simulation_manager):
    # initialize vehicle states
    vehicles.states[0:3] = simulation_manager.initial_pos  # NED
    vehicles.states[3:6] = simulation_manager.initial_vel  # NED
    vehicles.states[6:10] = euler_to_quaternion(simulation_manager.initial_attitude)
    vehicles.states[10:] = simulation_manager.initial_attitude_rate

    if vehicles.mass_changed or vehicles.inertia_changed:
        vehicles.update_inertia_and_mass()
        vehicles.move_origin_to_center_of_mass()

def run_simulation(vehicles, control_manager, environment_manager, simulation_manager):
    while simulation_manager.simulate:
        environment_manager.update(vehicles, simulation_manager)
        control_manager.update(vehicles, simulation_manager)
        vehicles.update_vehicle(control_manager, environment_manager, simulation_manager)
        vehicles.previous_states = copy.deepcopy(vehicles.states)
        force_b, moment_b = vehicles.calculate_loads_body_frame(control_manager, environment_manager,
                                                               simulation_manager)
        angular_momentum_be_b = vehicles.calculate_angular_momentum_body_frame()

        inertia_b = vehicles.inertia
        inv_inertia_b = vehicles.inv_inertia
        local_level_transform = vehicles.local_level_transform
        body_transform = vehicles.body_transform

        gravity_ll = environment_manager.gravity(vehicles)
        gravity_b = body_transform @ gravity_ll

        omega_be_b = vehicles.omega_be_b

        accel_b = force_b / vehicles.mass + gravity_b #- omega_be_b @ (body_transform @ vehicles.velocity) # TODO figure out later

        ang_accel_b = inv_inertia_b @ (
                -omega_be_b @ (inertia_b @ vehicles.ang_rate + angular_momentum_be_b) + moment_b)

        vehicles.states[3:6] = vehicles.states[3:6] + simulation_manager.dt * local_level_transform @ accel_b
        vehicles.states[10:13] = vehicles.states[10:13] + simulation_manager.dt * ang_accel_b

        vehicles.states[0:3] = vehicles.states[0:3] + simulation_manager.dt * vehicles.states[
                                                                            3:6] + 0.5 * simulation_manager.dt ** 2 * local_level_transform @ accel_b

        lam = 1 - (np.sum(vehicles.states[6:10] ** 2))
        k = 0.5
        vehicles.states[6:10] = vehicles.states[
                               6:10] + 0.5 * simulation_manager.dt * vehicles.att_rate_skew_matrix @ vehicles.states[
                                                                                                    6:10] + k * lam * vehicles.states[
                                                                                                                      6:10]
        vehicles.states[6:10] = vehicles.states[6:10] / np.linalg.norm(vehicles.states[6:10], ord=2)

        simulation_manager.update(vehicles, control_manager, environment_manager)
        if vehicles.mass_changed or vehicles.inertia_changed:
            vehicles.update_inertia_and_mass()
    print('Simulation finished.')

def simulate_vehicle(vehicles, control_manager, environment_manager, simulation_manager, visualizer=None):
    """

    :param Vehicle vehicle:
    :param control_manager:
    :param environment_manager:
    :param SimulationManager simulation_manager:
    :return:
    """
    initialize_simulation(vehicles, control_manager, environment_manager, simulation_manager)
    if visualizer is not None:
        sim_tread = Thread(target=run_simulation, args=(vehicles, control_manager, environment_manager, simulation_manager))
        sim_tread.start()
        visualizer(vehicles, simulation_manager, simulation_manager.rate)
    else:
        run_simulation(vehicles, control_manager, environment_manager, simulation_manager)






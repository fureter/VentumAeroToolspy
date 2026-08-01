import copy
from threading import Thread

import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from vispy import app

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

def state_space_update(states, dt, force_b, mass, gravity_b, inertia_b, inv_inertia_b, ang_rate, omega_be_b, angular_momentum_be_b, moment_b,
                       local_level_transform, att_rate_skew_matrix):
    accel_b = force_b / mass + gravity_b - omega_be_b @ (local_level_transform.T @ states[3:6]) # TODO figure out later

    ang_accel_b = inv_inertia_b @ (
            -omega_be_b @ (inertia_b @ ang_rate + angular_momentum_be_b) + moment_b)

    states[3:6] = states[3:6] + dt * local_level_transform @ accel_b
    states[10:13] = states[10:13] + dt * ang_accel_b

    states[0:3] = states[0:3] + dt * states[
        3:6] + 0.5 * dt ** 2 * local_level_transform @ accel_b

    lam = 1 - (np.sum(states[6:10] ** 2))
    k = 0.5
    states[6:10] = states[
                       6:10] + 0.5 * dt * att_rate_skew_matrix @ states[
                       6:10] + k * lam * states[
                       6:10]
    states[6:10] = states[6:10] / np.linalg.norm(states[6:10], ord=2)
    return states

def run_simulation(vehicles, control_manager, environment_manager, simulation_manager):
    while simulation_manager.simulate:
        environment_manager.update(vehicles, simulation_manager)
        control_manager.update(vehicles, simulation_manager)
        vehicles.update_vehicle(control_manager, environment_manager, simulation_manager)
        vehicles.previous_states = copy.deepcopy(vehicles.states)
        force_b, moment_b = vehicles.calculate_loads_body_frame(control_manager, environment_manager,
                                                               simulation_manager)
        angular_momentum_be_b = vehicles.calculate_angular_momentum_body_frame()

        inertia_b = vehicles.inertia_tensor
        inv_inertia_b = vehicles.inv_inertia
        local_level_transform = vehicles.local_level_transform
        body_transform = vehicles.body_transform

        gravity_ll = environment_manager.gravity(vehicles)
        gravity_b = body_transform @ gravity_ll

        omega_be_b = vehicles.omega_be_b

        vehicles.states = state_space_update(copy.deepcopy(vehicles.states), simulation_manager.dt, force_b,
                                             vehicles.mass, gravity_b, inertia_b, inv_inertia_b, vehicles.ang_rate,
                                             omega_be_b, angular_momentum_be_b, moment_b, local_level_transform,
                                             vehicles.att_rate_skew_matrix)


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






import copy

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from utilities.coordinate_systems import euler_to_quaternion
from vehicles import vehicle
from vehicles.vehicle import Vehicle
from simulation.simluation import SimulationManager

def simulate_vehicle(vehicle, control_manager, environment_manager, simulation_manager, visualizer=None):
    """

    :param Vehicle vehicle:
    :param control_manager:
    :param environment_manager:
    :param SimulationManager simulation_manager:
    :return:
    """
    # initialize vehicle states
    vehicle.states[0:3] = simulation_manager.initial_pos # NED
    vehicle.states[3:6] = simulation_manager.initial_vel # NED
    vehicle.states[6:10] = euler_to_quaternion(simulation_manager.initial_attitude)
    vehicle.states[10:] = vehicle.euler_to_body_rate_transform @ simulation_manager.initial_attitude_rate

    if vehicle.mass_changed or vehicle.inertia_changed:
        vehicle.update_inertia_and_mass()
        vehicle.move_origin_to_center_of_mass()

    def callback(intervals):
        ind = 0
        while simulation_manager.simulate and ind < intervals:
            environment_manager.update(vehicle, simulation_manager)
            control_manager.update(vehicle, simulation_manager)
            vehicle.control_input(control_manager)
            vehicle.previous_states = copy.deepcopy(vehicle.states)
            force_b, moment_b = vehicle.calculate_loads_body_frame(control_manager, environment_manager,
                                                                   simulation_manager)
            angular_momentum_be_b = vehicle.calculate_angular_momentum_body_frame()

            inertia_b = vehicle.inertia
            inv_inertia_b = vehicle.inv_inertia
            local_level_transform = vehicle.local_level_transform
            body_transform = vehicle.body_transform

            gravity_ll = environment_manager.gravity(vehicle)
            gravity_b = body_transform @ gravity_ll

            omega_be_b = vehicle.omega_be_b

            accel_b = force_b / vehicle.mass + gravity_b #- omega_be_b @ (body_transform @ vehicle.velocity) TODO figure out later

            ang_accel_b = inv_inertia_b @ (
                        -omega_be_b @ (inertia_b @ vehicle.ang_rate + angular_momentum_be_b) + moment_b)

            vehicle.states[3:6] = vehicle.states[3:6] + simulation_manager.dt * local_level_transform @ accel_b
            vehicle.states[10:13] = vehicle.states[10:13] + simulation_manager.dt * ang_accel_b

            vehicle.states[0:3] = vehicle.states[0:3] + simulation_manager.dt * vehicle.states[
                3:6] + 0.5 * simulation_manager.dt ** 2 * local_level_transform @ accel_b

            lam = 1 - (np.sum(vehicle.states[6:10]**2))
            k = 0.5
            vehicle.states[6:10] = vehicle.states[6:10] + 0.5 * simulation_manager.dt * vehicle.att_rate_skew_matrix @ vehicle.states[6:10] + k*lam*vehicle.states[6:10]
            vehicle.states[6:10] = vehicle.states[6:10] / np.linalg.norm(vehicle.states[6:10], ord=2)

            simulation_manager.update(vehicle, control_manager, environment_manager)
            if vehicle.mass_changed or vehicle.inertia_changed:
                vehicle.update_inertia_and_mass()
            ind += 1

    if visualizer is not None:
        visualizer(vehicle, simulation_manager, simulation_manager.rate, callback)
    else:
        callback(simulation_manager.rate * simulation_manager.max_runtime)






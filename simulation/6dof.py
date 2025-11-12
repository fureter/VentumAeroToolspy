import numpy as np
import scipy

from vehicles import vehicle
from vehicles.vehicle import Vehicle
from simulation.simluation import SimulationManager

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

    simulation_manager.logger.log_interation(vehicle,control_manager,environment_manager, simulation_manager)

    while(simulation_manager.simulate):
        environment_manager.update(vehicle, simulation_manager)
        control_manager.update(vehicle, simulation_manager)
        force_b, moment_b = vehicle.calculate_loads_body_frame(control_manager, environment_manager, simulation_manager)
        angular_momentum_be_b = vehicle.calculate_angular_momentum_body_frame()


        inertia_b = vehicle.inertia
        inv_inertia_b = vehicle.inv_inertia
        local_level_transform = vehicle.local_level_transform

        gravity_ll = environment_manager.gravity(vehicle)
        gravity_b = local_level_transform * gravity_ll

        omega_be_b = vehicle.omega_be_b

        velocity_b = local_level_transform * vehicle.velocity
        vel_det = vehicle.mass * np.linalg.det(np.array([[1, 1, 1],
                                           [vehicle.states[9],vehicle.states[10],vehicle.states[11]],
                                           [velocity_b[0],velocity_b[1],velocity_b[2]]]))

        accel_b = force_b/vehicle.mass - vel_det + gravity_b
        ang_accel_b = inv_inertia_b * (-omega_be_b * (inertia_b * vehicle.ang_rate  +  angular_momentum_be_b) + moment_b)

        vehicle.states[3:6] = vehicle.states[3:6] + simulation_manager.dt * local_level_transform.T * accel_b
        vehicle.states[9:12] = vehicle.states[9:12] + simulation_manager.dt * ang_accel_b
        simulation_manager.update(vehicle, control_manager, environment_manager, simulation_manager)
        simulation_manager.logger.log_iteration(vehicle, control_manager, environment_manager, simulation_manager)




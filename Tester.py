import logging
import sys
import os

import matplotlib
import matplotlib.pyplot as plt

from gnc.control_manager import RateController, ControlManager, AttitudeController
from simulation.simluation import SimulationManager, EnvironmentManager
from utilities.coordinate_systems import rotation_matrix
from vehicles.projectile import SimpleDart, SimpleWingedDart

# matplotlib.use('AGG')
import numpy as np

from acoustics.emit_receive import Emitter
from simulation.acoustic_sim import simulate_2d_static_emitter_field
from simulation.dof6 import simulate_vehicle
from vehicles.components.simple_components import HollowCylinder, SolidSphere, RectangularPrism, ThinPlate
from visualize.primative_visualize import plot_vehicle


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # artillery = SimpleDart(name='100mm', tail_roll=0, tail_chord=0.4, tail_span=0.6)
    artillery = SimpleWingedDart(name='100mm', tail_roll=0, tail_chord=0.4, tail_span=0.6)

    rotation_matrix([np.deg2rad(90),np.deg2rad(45),np.deg2rad(0)])
    launch_angle = np.deg2rad(30)
    roll_offset = np.deg2rad(-15)
    yaw_offset = np.deg2rad(0)
    pitch_offset = np.deg2rad(0)
    launch_vel=50
    simulation_manager = SimulationManager(pos_init=np.array([0,0,0]),
                                           vel_init=np.array([launch_vel*np.cos(launch_angle),0,-launch_vel*np.sin(launch_angle)]),
                                           angle_init=np.array([roll_offset,launch_angle+pitch_offset,yaw_offset]),
                                           angle_rate_init=np.array([0.0,0.0,0.0]), log=logger,
                                           sim_rate=100, max_runtime=50)
    control_manager = AttitudeController() #ControlManager() #RateController()
    environment_manager = EnvironmentManager()
    simulate_vehicle(vehicle=artillery, control_manager=control_manager,
                     environment_manager=environment_manager, simulation_manager=simulation_manager, visualizer=plot_vehicle)
    simulation_manager.plot_velocity()
    simulation_manager.plot_position()
    simulation_manager.plot_angle()
    simulation_manager.plot_angle_rate()
    simulation_manager.plot_force()
    simulation_manager.plot_moment()
    artillery.plot_tail_forces(simulation_manager)
    plt.show()

    # plot_vehicle(vehicle=artillery, simulation_manager=simulation_manager)

    # freq = 25E3
    # speed_of_sound = 337
    # wave_length = speed_of_sound / freq
    # phase_deltas = list(range(0,10))
    # num_emitters = 2
    # emitters = list()
    # for ind in range(num_emitters):
    #     x = (ind - num_emitters/2) * wave_length/2
    #     emitters.append(Emitter(init_position=np.array([x,-2*wave_length,0]), init_attitude=np.array([0,0,0]), emit_power=30, frequency=freq,
    #                       antenna_pattern=None))
    # for phase_delta in phase_deltas:
    #     phase_offset = list()
    #     deg_shift = phase_delta/num_emitters
    #     for ind in range(num_emitters):
    #         phase_offset.append(ind * deg_shift)
    #     bb = np.array([[-20*wave_length,20*wave_length],[-4*wave_length,200*wave_length]])
    #     simulate_2d_static_emitter_field(emitters=emitters, bounding_box=bb, spatial_resolution=wave_length/4,
    #                                      phase_offsets=phase_offset, output_dir=os.path.dirname(__file__), title='%semitters_%s_phase_delta' % (num_emitters, phase_delta))

    #
    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.INFO)
    # logger.addHandler(handler)
    #
    # logging.getLogger('matplotlib.font_manager').disabled = True
    #
    # directory = r"M:\Projects\OpenFOAM\AE410Midterm"
    #
    # t = OpenFOAMUtils.get_time_steps(directory)
    # rho = OpenFOAMUtils.load_from_directory('rho', directory, grid_size=150)
    # p = OpenFOAMUtils.load_from_directory('p', directory, grid_size=150)
    # u = OpenFOAMUtils.load_from_directory('U', directory, grid_size=150, scalar=False)
    #
    # legend = list()
    # for time in t:
    #     legend.append('t = %sms' % (time*1E3))
    #
    # x_length = 10.0
    # x_resolution = 150
    # x_grid = np.linspace(0, x_length, num=x_resolution, endpoint=False)
    #
    # rho_initial = np.ones(x_resolution) * 1.225
    # u_initial = np.ones(x_resolution) * 100
    # p_initial = 101325 * (1 + 1/20 * np.exp(-10*(x_grid - x_length/2)**2))
    #
    # rho[0,:] = rho_initial
    # u[0,:] = u_initial
    # p[0,:] = p_initial
    #
    # plt.subplot(3,1,1)
    # plt.grid(True, which='both')
    # for ind in range(len(t)):
    #     plt.plot(x_grid,rho[ind,:])
    # plt.legend(legend)
    # plt.title('Density (kg/m^3)')
    # plt.subplot(3,1,2)
    # plt.grid(True, which='both')
    # for ind in range(len(t)):
    #     plt.plot(x_grid,u[ind,:])
    # plt.legend(legend)
    # plt.title('Fluid Velocity (m/s)')
    # plt.subplot(3,1,3)
    # plt.grid(True, which='both')
    # for ind in range(len(t)):
    #     plt.plot(x_grid,p[ind,:])
    # plt.legend(legend)
    # plt.title('Pressure (Pa)')
    # plt.show()

if "__main__" in __name__:
    main()

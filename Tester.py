import logging
import sys
import os
import timeit

import matplotlib
import matplotlib.pyplot as plt

from aerodynamics.airfoil import NerualAirfoil
from gnc.control_manager import RateController, ControlManager, AttitudeController
from waveforms.modulation import Waveform
from simulation.blade_element import BladeElementMethod
from simulation.simluation import SimulationManager, EnvironmentManager
from utilities.coordinate_systems import frd_dcm
from utilities.data_logging import DataLogger
from vehicles.components.Propeller import Propeller
from vehicles.projectile import SimpleDart, SimpleWingedDart, DeployableWingedDart

# matplotlib.use('AGG')
import numpy as np
from sympy.printing.latex import LatexPrinter, print_latex
import sympy as sy

from acoustics.emit_receive import Emitter
from simulation.acoustic_sim import simulate_2d_static_emitter_field
from simulation.dof6 import simulate_vehicle
from vehicles.components.simple_components import HollowCylinder, SolidSphere, RectangularPrism, ThinPlate
from visualize.primative_visualize import plot_vehicle
from visualize.signal_plotting import plot_waveform_fft, plot_waveform_periodgram


def main():
    logger = DataLogger()
    # time = 100E-6
    # base_freq = 1.023E6
    # freq = base_freq * 100
    # carrier_freq = base_freq * 10
    # chipping_freq = base_freq * 5
    # waveform = Waveform.boc(np.linspace(0.0, time, int(time*freq), endpoint=True), 1.0/freq,
    #                         carrier_freq, chipping_freq)
    # plot_waveform_fft(waveform, show=False)
    # plot_waveform_periodgram(waveform, show=True)
    # test_prop = Propeller(mass=0, position=np.array([0.0, 0.0, 0.0]), angle=np.array([0.0, 0.0, 0.0]),
    #                       inertia_0=np.array([0.0, 0.0, 0.0]), num_id=0)
    # segment_chords = np.linspace(0.04, 0.025, 10, endpoint=True)
    # segment_twist = np.linspace(np.deg2rad(60), np.deg2rad(25.0), 10, endpoint=True)
    # segment_spans = np.linspace(0.005, 0.2, 10, endpoint=True)
    # af = NerualAirfoil('NACA4412', re_list=np.linspace(1000,600000, 360), aoa_list=np.linspace(-180, 180, 360, endpoint=True))
    # airfoils = [af]*10
    # test_prop.initialize(airfoils, segment_chords, segment_spans, segment_twist, 2)
    # bem  = BladeElementMethod(test_prop)
    # rpm_range = np.linspace(1000,10000, 32, endpoint=True)
    # vel_range = np.linspace(0,100, 11, endpoint=True)
    # rho_range = np.array([1.225])
    # bem.analyze_propeller(vel_range, rpm_range, rho_range)
    # plt.figure()
    # leg = []
    # for vel_idx in range(len(vel_range)):
    #     plt.plot(rpm_range, test_prop.thrust_curves[vel_idx,:,0])
    #     leg.append('Vel=%sm/s' % vel_range[vel_idx])
    # plt.legend(leg)
    # plt.grid()
    # plt.xlabel('rpm')
    # plt.ylabel('thrust (N)')
    #
    # plt.figure()
    # leg = []
    # for vel_idx in range(len(vel_range)):
    #     plt.plot(rpm_range, test_prop.torque_curves[vel_idx,:,0])
    #     leg.append('Vel=%sm/s' % vel_range[vel_idx])
    # plt.legend(leg)
    # plt.grid()
    # plt.xlabel('rpm')
    # plt.ylabel('Torque (Nm)')
    # plt.show()
    # print('yay')
    # artillery = SimpleDart(name='100mm', tail_roll=0, tail_chord=0.4, tail_span=0.6)
    # artillery = SimpleWingedDart(name='100mm', tail_roll=0, tail_chord=0.4, tail_span=0.6)
    # artillery = DeployableWingedDart(name='100mm', tail_chord=0.2, tail_span=0.8, data_logger=logger)
    #
    # zenith = np.array([0, 0, -1.0])
    # dcm = frd_dcm(np.array([np.deg2rad(0), np.deg2rad(90), np.deg2rad(0)]))
    # print(dcm.T @ zenith)
    # quit()
    # launch_angle = np.deg2rad(60)
    # roll_offset = np.deg2rad(0)
    # yaw_offset = np.deg2rad(-90)
    # pitch_offset = np.deg2rad(0)
    # launch_vel=20
    # simulation_manager = SimulationManager(pos_init=np.array([0,0,0]),
    #                                        vel_init=np.array([launch_vel*np.cos(launch_angle)*np.cos(yaw_offset),
    #                                                           launch_vel*np.cos(launch_angle)*np.sin(yaw_offset),
    #                                                           -launch_vel*np.sin(launch_angle)]),
    #                                        angle_init=np.array([roll_offset,launch_angle+pitch_offset,yaw_offset]),
    #                                        angle_rate_init=np.array([0.0,0.0,0.0]), log=logger,
    #                                        sim_rate=50, max_runtime=20.0)
    # control_manager = AttitudeController(logger, update_rate=50.0) #ControlManager() #RateController()
    # environment_manager = EnvironmentManager(data_logger=logger)
    # start = timeit.default_timer()
    # simulate_vehicle(vehicles=artillery, control_manager=control_manager,
    #                  environment_manager=environment_manager, simulation_manager=simulation_manager, visualizer=None)#plot_vehicle)
    # print('Run time: %s' % (timeit.default_timer() - start))
    # start = timeit.default_timer()
    # simulation_manager.generate_all_plots()
    # artillery.plot_default_parameters()
    # artillery.plot_tail_forces()
    # plt.show()

    # plot_vehicle(vehicle=artillery, simulation_manager=simulation_manager)

    freq = 25E3
    speed_of_sound = 337.0
    wave_length = speed_of_sound / freq
    phase_deltas = list(range(45,46))
    num_emitters = 20
    emitters = list()
    for ind in range(num_emitters):
        x = (ind - num_emitters/2) * wave_length/2
        emitters.append(Emitter(init_position=np.array([x,-2*wave_length,0]), init_attitude=np.array([0,0,0]), emit_power=30, frequency=freq,
                          antenna_pattern=None))
    for phase_delta in phase_deltas:
        phase_offset = list()
        deg_shift = phase_delta
        for ind in range(num_emitters):
            dt = wave_length/2.0 * np.sin(np.deg2rad(deg_shift) / speed_of_sound)
            phase_offset.append(ind * dt)
        bb = np.array([[-50*wave_length,50*wave_length],[-50*wave_length,50*wave_length]])
        simulate_2d_static_emitter_field(emitters=emitters, bounding_box=bb, spatial_resolution=wave_length/4,
                                         phase_offsets=phase_offset, output_dir=os.path.dirname(__file__), title='%semitters_%sdeg_phase_delta' % (num_emitters, phase_delta))

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

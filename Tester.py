import logging
import sys
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')
import numpy as np

from acoustics.emit_receive import Emitter
from simulation.acoustic_sim import simulate_2d_static_emitter_field


def main():
    freq = 25E3
    speed_of_sound = 337
    wave_length = speed_of_sound / freq
    phase_deltas = list(range(0,10))
    num_emitters = 2
    emitters = list()
    for ind in range(num_emitters):
        x = (ind - num_emitters/2) * wave_length/2
        emitters.append(Emitter(init_position=np.array([x,-2*wave_length,0]), init_attitude=np.array([0,0,0]), emit_power=30, frequency=freq,
                          antenna_pattern=None))
    for phase_delta in phase_deltas:
        phase_offset = list()
        deg_shift = phase_delta/num_emitters
        for ind in range(num_emitters):
            phase_offset.append(ind * deg_shift)
        bb = np.array([[-20*wave_length,20*wave_length],[-4*wave_length,200*wave_length]])
        simulate_2d_static_emitter_field(emitters=emitters, bounding_box=bb, spatial_resolution=wave_length/4,
                                         phase_offsets=phase_offset, output_dir=os.path.dirname(__file__), title='%semitters_%s_phase_delta' % (num_emitters, phase_delta))
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
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
import copy
import logging
import sys
import os
import timeit

import matplotlib
import matplotlib.pyplot as plt
from scipy import constants
import numpy as np
from torch.ao.quantization import convert_jit

# from aerodynamics.airfoil import NerualAirfoil
from analysis.aircraft_performance import SimplifiedLongitudinal, simplified_longitudinal_example_elevator_doublet, \
    simplified_longitudinal_example_elevator_pulse, simplified_longitudinal_example_unobservable, \
    simplified_longitudinal_example_thrust_3211, \
    simplified_longitudinal_example_elevator_multimaneuver
# from gnc.control_manager import RateController, ControlManager, AttitudeController
import signals
from signals.waveform import Waveform
from visualize import signal_plotting
# from simulation.blade_element import BladeElementMethod
# from simulation.simluation import SimulationManager, EnvironmentManager
# from utilities.coordinate_systems import frd_dcm
# from utilities.data_logging import DataLogger
# from vehicles.components.Propeller import Propeller
# from vehicles.projectile import SimpleDart, SimpleWingedDart, DeployableWingedDart

# matplotlib.use('AGG')
import numpy as np

from analysis.parameter_estimator import ParameterEstimator
from antennas.antenna_pattern import AntennaPattern


# from sympy.printing.latex import LatexPrinter, print_latex
# import sympy as sy

# from acoustics.emit_receive import Emitter
# from simulation.acoustic_sim import simulate_2d_static_emitter_field
# from simulation.dof6 import simulate_vehicle
# from vehicles.components.simple_components import HollowCylinder, SolidSphere, RectangularPrism, ThinPlate
# from visualize.primative_visualize import plot_vehicle
# from visualize.signal_plotting import plot_waveform_fft, plot_waveform_periodgram


def main():
    # wavelen = 342/200E3
    # elem_position = np.array([[1.0, 0.0, 0.0],[0.0,1.0,0.0],[-1.0,0.0,0.0], [0.0,-1.0,0.0]]) * wavelen/2
    # freq = np.linspace(100E3, 200E3, 2, endpoint=True)
    # az_res = np.deg2rad(4)
    # el_res = np.deg2rad(2)
    # antenna = AntennaPattern.gen_ideal_omni_array(elem_position=elem_position, frequency_range=freq, az_resolution=az_res,
    #                                               el_resolution=el_res, elem_clocking=None, wavespeed=342.0)
    # antenna.plot_gain_hemispheres(freq=60E3)
    # antenna.plot_phase_hemispheres(freq=200E3, show=True)
    # antenna.plot_azimuth_slice(show=True)
    BLOCK_TIME = 10E-3
    BLOCKS = int(10**(0.0/10))
    f_ref = 1.023E6
    chipping_freq = f_ref*10
    chipping_time = 1/chipping_freq
    carrier_freq = 1575.42E6

    sample_time = BLOCK_TIME
    sample_freq = 40E6
    samples = int(sample_time*sample_freq)

    prn_codes = list()
    for prn, taps in {1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9)}.items():
        prn_codes.append(signals.waveform.gps_ca(*taps))
    prn_codes = np.array(prn_codes)

    bit_sequence = np.squeeze(np.repeat(prn_codes[1], int(BLOCK_TIME/chipping_time)))
    bit_sequence[bit_sequence == 0] = -1

    chipping_rate = 1/chipping_freq
    chip_scan = np.linspace(-10, 10, 21, endpoint=True)
    doppler_scan = np.linspace(-10000, 10000, 11, endpoint=True)
    correlation = np.zeros([chip_scan.shape[0], doppler_scan.shape[0]], dtype=np.complex128)
    mid_length = samples//2

    def get_bits(mid_point, total_samples, bits):
        mid_point = mid_point % len(bits)
        if mid_point - total_samples/2 < 0:
            remainder = total_samples + int(mid_point - total_samples/2)
            return np.append(bits[int(mid_point - total_samples/2):], bits[0:remainder])
        elif int(mid_point + total_samples/2) > len(bits):
            remainder = (mid_point + total_samples) - total_samples
            return np.append(bits[int(mid_point - total_samples/2):], bits[0:remainder])
        else:
            return bits[int(mid_point - total_samples/2):int(mid_point + total_samples/2)]

    for block in range(BLOCKS):
        block_start = block * int(BLOCK_TIME*sample_freq) + samples//2 + mid_length
        for chip_ind, chip in enumerate(chip_scan):
            print('Chip: %s' % chip_ind)
            signal = Waveform.bpsk(samples=samples, sampling_freq=sample_freq,
                                   carrier_freq=carrier_freq, chipping_freq=chipping_freq,
                                   bit_sequence=get_bits(block_start, samples, bit_sequence))
            signal.adjust_power(0.0)
            replica_offset = int(chip * chipping_rate * sample_freq)
            replica_bits1 = get_bits(block_start + replica_offset, samples, bit_sequence)
            replica_signal = Waveform.bpsk(samples=samples, sampling_freq=sample_freq,
                                          carrier_freq=carrier_freq,
                                          chipping_freq=chipping_freq,
                                          bit_sequence=replica_bits1)
            for dop_ind, doppler in enumerate(doppler_scan):
                noise = Waveform.awgn(time=sample_time, sampling_frequency=sample_freq, center_freq=10E6, bw=20E6,
                                      power=0)
                replica_signal.freq_shift(doppler)
                integration_wfm = copy.deepcopy(signal)
                integration_wfm.data = np.zeros(integration_wfm.samples, dtype=np.complex128)
                integration_wfm.data += (signal).data / sample_time
                correlation[chip_ind, dop_ind] += integration_wfm.correlation(replica_signal)
                replica_signal.freq_shift(-doppler)

    correlation = 10.0*np.log10(correlation)

    chip_mesh, dop_mesh = np.meshgrid(chip_scan, doppler_scan)
    plt.figure()
    if doppler_scan.shape[0] > 1:
        plt.contourf(chip_mesh, dop_mesh, correlation.T, levels=64)
        plt.xlabel('Code Offset (Chips)')
        plt.ylabel('Frequency Offset (Hz)')
        plt.colorbar()
    else:
        plt.plot(chip_scan, correlation[:,0])
        plt.xlabel('Code Offset (Chips)')
        plt.ylabel('Correlation')
        plt.grid(True)
    plt.show()






if "__main__" in __name__:
    main()

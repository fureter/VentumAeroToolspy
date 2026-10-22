import copy
import logging
import sys
import os
import timeit

import matplotlib
import matplotlib.pyplot as plt

# from aerodynamics.airfoil import NerualAirfoil
from analysis.aircraft_performance import SimplifiedLongitudinal, simplified_longitudinal_example_elevator_doublet, \
    simplified_longitudinal_example_elevator_pulse, simplified_longitudinal_example_unobservable, \
    simplified_longitudinal_example_thrust_3211, \
    simplified_longitudinal_example_elevator_multimaneuver
# from gnc.control_manager import RateController, ControlManager, AttitudeController
# from waveforms.modulation import Waveform
# from simulation.blade_element import BladeElementMethod
# from simulation.simluation import SimulationManager, EnvironmentManager
# from utilities.coordinate_systems import frd_dcm
# from utilities.data_logging import DataLogger
# from vehicles.components.Propeller import Propeller
# from vehicles.projectile import SimpleDart, SimpleWingedDart, DeployableWingedDart

# matplotlib.use('AGG')
import numpy as np

from analysis.parameter_estimator import ParameterEstimator


# from sympy.printing.latex import LatexPrinter, print_latex
# import sympy as sy

# from acoustics.emit_receive import Emitter
# from simulation.acoustic_sim import simulate_2d_static_emitter_field
# from simulation.dof6 import simulate_vehicle
# from vehicles.components.simple_components import HollowCylinder, SolidSphere, RectangularPrism, ThinPlate
# from visualize.primative_visualize import plot_vehicle
# from visualize.signal_plotting import plot_waveform_fft, plot_waveform_periodgram


def main():
    simplified_longitudinal_example_thrust_3211()
    simplified_longitudinal_example_elevator_multimaneuver()
    simplified_longitudinal_example_elevator_doublet()
    simplified_longitudinal_example_elevator_pulse()
    simplified_longitudinal_example_unobservable()


if "__main__" in __name__:
    main()

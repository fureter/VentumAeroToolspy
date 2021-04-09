import logging
import sys

from Design.Components.Wing import Wing
from Design.Aircraft import Aircraft

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logging.getLogger('matplotlib.font_manager').disabled = True

    span = [0.1, 0.9, 0.2, 0.1]
    chord = [0.2, 0.2, 0.14, 0.1, 0.05]
    twist = 0
    sweep = [3, 5, 10, 45]
    polyhedral = 1

    span_h = [0.2, 0.1]
    chord_h = [0.14, 0.10, 0.06]
    twist_h = 0
    sweep_h = [10]
    polyhedral_h = 1

    wing = Wing(sub_type=Wing.SubType.MAIN_WING, span=span, chord=chord, twist=twist, sweep=sweep,
                polyhedral=polyhedral, logger=logger)

    horz_stab = Wing(sub_type=Wing.SubType.HORZ_STAB, span=span_h, chord=chord_h, twist=twist_h, sweep=sweep_h,
                polyhedral=polyhedral_h, logger=logger)

    logger.info('Wing Area: %sm²' % wing.area)
    logger.info('Wing Projected Area: %sm²' % wing.projected_area)
    logger.info('Wing MAC: %sm' % wing.mac)
    logger.info('Wing AR: %s' % wing.ar)

    aircraft = Aircraft(name='Yolo Plane', logger=logger)
    aircraft.add_component(wing, (0,0,0))
    aircraft.add_component(horz_stab, (-1.2,0,0))

    Aircraft.plot_planform(aircraft)

main()
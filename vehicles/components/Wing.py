import logging
import numpy as np
from matplotlib import pyplot as plt

from .Components import Component
from aerodynamics.airfoil import Airfoil


class Wing(Component):

    def __init__(self, sub_type, span, chord, sweep, polyhedral, symmetric=True, twist=None, Airfoil=None, logger=None):
        """
        Wings are formed from various segments, with each segment being a portion of the lists given to for __init__.
        A wing can be formed from a single segment or various segments, and each input must either be a single value
        that defines every segment of the wing, a list with length one that defines every segment of the wing, or a list
        with length n, where n is the number of segments, and each value corresponds to a segment.

        e.x: Wing(Wing.SubType.HORZ_STAB, span=(0.2,0.1), chord=(0.1, 0.08, 0.4), sweep=0, polyhedral=(2))
            This will generate a wing that is meant to represent the horizontal stabilizer, that is a total length of
            60cm, due to it having a segment of length 20cm and a segment of length 10cm and is symmetric. The wing will
            have a root chord of 10cm, a middle chord of 8cm at 20cm from the root, and a tip chord of 4cm. The sweep of
            the wing will be 0° at the quarter chord line. Finally the wing will have a dihedral of 2° due to there
            being a single polyhedral value given, making each segment have a polyhedral of 2°.

        :param sub_type: Enum that tracks what the wings purpose it. This is used to simplify design by making it easier
        to match wings to the equations.
        :param span: list of lengths of each wing segment. List is length n, where n is the number of segments
        :param chord: list of chord length values for each wing segment, list is length n+1 where n is the number of
        segments.
        :param sweep: list of absolute sweep values for the quater chord of each wing segment. Length n.
        :param polyhedral: list of values defining the absolute polyhedral of each wing segment. Length n.
        :param symmetric: bool to define whether the wing is symmetric or a single wing section. Symmetry is assumed
        to be about the xz plane.
        :param twist: list of absolute twist values for the wing cross-section, with negative values being wash out.
        :param Airfoil: list of Airfoil objects to define the cross section of the wing. This is not a required field
        for initial design, but is required for future analysis. Length n+1
        :param logger: Logger instance for printing data to the terminal, if not given the default Python instance is
        taken.
        """

        super().__init__()

        if logger is None:
            logger = logging.getLogger()
        self._logger = logger

        self.type = 'Wing'
        self.sub_type = sub_type
        self.span = span
        self.chord = chord
        self.sweep = sweep
        self.polyhedral = polyhedral
        self.twist = twist
        self.airfoil = Airfoil
        self.symmetric = symmetric

        self._standardize_data_types()

        self.area = self._area
        self.projected_area = self._projected_area
        self.mac = self._mean_aerodynamic_chord
        self.ar = self._aspect_ratio


    def total_span(self):
        span = np.sum(self.span)
        span = span*2 if self.symmetric else span
        return span


    def _standardize_data_types(self):

        if isinstance(self.span, int):
            self._logger.debug('Integer Span given, converting to single value list (span)')
            self.span = [self.span]

        n = len(self.span)

        if isinstance(self.chord, int):
            self._logger.debug('Integer Chord given, converting to list size (%s) (chord,...., chord)' % (n+1))
            self.chord = [self.chord] * (len(self.span)+1)

        if isinstance(self.sweep, int):
            self._logger.debug('Integer Sweep given, converting to list size (%s)' % n)
            self.sweep = [self.sweep]* len(self.span)

        if isinstance(self.polyhedral, int):
            self._logger.debug('Integer Polyhedral given, converting to list size (%s)' % n)
            self.polyhedral = [self.polyhedral]* len(self.span)

        if isinstance(self.twist, int):
            self._logger.debug('Integer Twist given, converting to list size (%s)' % n)
            self.twist = [self.twist] * len(self.span)

        len_c = len(self.chord)
        len_s = len(self.sweep)
        len_p = len(self.polyhedral)
        len_t = len(self.twist)

        if (len_c) != (n + 1):
            self._logger.exception('Size of Chord list is not n+1, cannot form wing :: %s != (%s + 1)' % (len_c, n))
            raise

        if len_s != n:
            if len_s == 1:
                self.sweep = self.sweep* len(self.span)
            else:
                self._logger.exception('Size of Sweep Sweep is not n, cannot form wing :: %s != (%s)' % (len_s, n))
                raise

        if len_p != n:
            if len_p == 1:
                self.polyhedral = self.polyhedral* len(self.span)
            else:
                self._logger.exception('Size of Polyhedral list is not n, cannot form wing :: %s != (%s)' % (len_p, n))
                raise

        if len_t != n:
            if len_t == 1:
                self.twist = self.twist* len(self.span)
            else:
                self._logger.exception('Size of Twist list is not n, cannot form wing :: %s != (%s)' % (len_t, n))
                raise

        if self.airfoil is not None:
            if isinstance(self.airfoil, Airfoil):
                self._logger.debug('Single Airfoil given, converting to (%s + 1) Airfoil list' % n)
                self.airfoil = [self.airfoil] * (len(self.span) +1)

            len_a = len(self.airfoil)

            if (len_a) != (n + 1):
                if len_a == 1:
                    self.airfoil = self.airfoil* (len(self.span) +1)
                else:
                    self._logger.exception('Size of Airfoil list is not n+1, cannot form wing :: %s != (%s + 1)' % (len_a, n))
                    raise


    @property
    def area(self):
        s = 0
        for inx in range(0, len(self.span)):
            s += self.span[inx]*(self.chord[inx] + self.chord[inx+1])/2

        s = s*2 if self.symmetric else s

        return s


    @property
    def projected_area(self):
        s = 0
        for inx in range(0, len(self.span)):
            s += (self.span[inx] * (self.chord[inx] + self.chord[inx + 1]) / 2) * np.cos(
                np.deg2rad(self.polyhedral[inx]))

        s = s * 2 if self.symmetric else s

        return s


    @property
    def mean_aerodynamic_chord(self):
        mac = self.area / self.total_span()

        return mac


    @property
    def aspect_ratio(self):
        ar = self.total_span()**2 / self.area
        return ar


    def planform_coordinates(self, half=False):
        len_c = len(self.chord)
        x_init = list()
        top_init = list()
        bottom_init = list()

        rel_x = 0
        rel_y = 0

        # Get the coordinates for the leading and trailing edge of the wing.
        for inx in range(0,len_c):
            x_init.append(rel_x)
            top_init.append(self.chord[inx] * 0.25 + rel_y)
            bottom_init.append(-self.chord[inx] * 0.75 + rel_y)

            if inx < len_c - 1:
                rel_x += self.span[inx]
                rel_y += self.span[inx]*np.sin(np.deg2rad(-self.sweep[inx]))

        x = list()
        y = list()

        for inx in range(0, len(x_init)):
            x.append(x_init[inx])
            y.append(top_init[inx])
        for inx in range(len(x_init)-1, 0, -1):
            x.append(x_init[inx])
            y.append(bottom_init[inx])
        if self.symmetric and half is False:
            for inx in range(0, len(x_init)):
                x.append(-x_init[inx])
                y.append(bottom_init[inx])
            for inx in range(len(x_init)-1, -1, -1):
                x.append(-x_init[inx])
                y.append(top_init[inx])

        return np.array(x), np.array(y)


    def plot_wing(self):
        (x, y) = self.planform_coordinates()

        plt.plot(x, y)
        plt.xlabel('y (m)')
        plt.ylabel('x (m)')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


    class SubType(object):
        MAIN_WING = 'Main Wing'
        HORZ_STAB = 'Horizontal Stabilizer'
        VERT_STAB = 'Vertical Stabilizer'

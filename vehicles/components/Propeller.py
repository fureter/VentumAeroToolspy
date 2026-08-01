import numpy as np

from vehicles.components.components import Component


class Propeller(Component):
    def __init__(self, mass, position, angle, inertia_0, num_id=0):
        super(Propeller, self).__init__(mass, position, angle, inertia_0, num_id)
        self.airfoil_list = []
        self.segment_chords = None
        self.segment_spans = None
        # Insert segement sweep later...
        self.segment_twist = None

        self.num_segments = None
        self.num_blade = None

        self.velocity_range = None
        self.rpm_range = None
        self.rho_range = None
        self.thrust_curves = None
        self.torque_curves = None

        self.valid_data = False

    def initialize(self, airfoils, segment_chords, segment_spans, segment_twist, num_blades):
        assert len(airfoils) == len(segment_chords) == len(segment_spans) == len(
            segment_twist), "Error: Segment definitions do not match in size"

        self.airfoil_list = airfoils
        self.segment_chords = np.array(segment_chords)
        self.segment_spans = np.array(segment_spans)
        self.segment_twist = np.array(segment_twist)
        self.num_segments = self.segment_chords.shape[0]-1
        self.num_blades = num_blades
import numpy as np
from astropy.coordinates.builtin_frames.itrs_observed_transforms import altaz_to_hadec_mat

from aerodynamics.aero_lookup_table import AeroTable
from utilities.coordinate_systems import rotation_matrix
from .components import  Component

class HollowCylinder(Component):
    def __init__(self, mass, outer_radius, inner_radius, length, position, angle):
        """
        X-Axis is along the long axis of the cylinder
        :param mass:
        :param outer_radius:
        :param inner_radius:
        :param length:
        :param position:
        :param angle:
        """
        inertia_0 = np.array([[0.5 * mass * (outer_radius**2 + inner_radius**2), 0.0, 0.0],
                               [0.0, 1.0/12.0 * mass * (3* (outer_radius**2 + inner_radius**2) + length**2), 0.0],
                               [0.0, 0.0, 1.0/12.0 * mass * (3* (outer_radius**2 + inner_radius**2) + length**2)]])
        super().__init__(mass, position, angle, inertia_0)

    def calculate_loads_body_frame(self, vehicle, control_manager, environment_manager, simulation_manager):
        return np.array([0.0, 0.0, 0.0]), np.array([0.00000, 0.00000, 0.0000])

    def calculate_angular_momentum_body_frame(self, body_rate):
            return np.zeros(3)

class SolidSphere(Component):
    def __init__(self, mass, radius, position, angle):
        inertia_0 = 2/5 * mass * radius**2 * np.eye(3)
        self.radius = radius
        super().__init__(mass, position, angle, inertia_0)

    def calculate_loads_body_frame(self, vehicle, control_manager, environment_manager, simulation_manager):
        body_velocity = vehicle.body_transform @ vehicle.velocity
        component_velocity = self.body_to_component_transform @ (body_velocity + np.cross(vehicle.ang_rate,  self.position - vehicle.center_of_mass))
        volume = 4/3 * np.pi * self.radius**3
        cd = 0.025
        rho = environment_manager.air_properties()[0]
        drag = -0.5 * np.linalg.norm(component_velocity,ord=2)**2 * cd * rho * np.pi*self.radius**2 * component_velocity / np.linalg.norm(component_velocity,ord=2)
        if np.any(np.isnan(drag)):
            drag = np.zeros(3)
        bouyancy = vehicle.body_transform @ np.array([0,0,rho * volume * -environment_manager.gravity(simulation_manager.lla[2])[2]])

        force =  bouyancy + self.body_to_component_transform.T @ drag
        moment = np.cross(self.position - vehicle.center_of_mass, force)

        return  force, moment

    def calculate_angular_momentum_body_frame(self, body_rate):
            return np.zeros(3)

class ThinPlate(Component):
        def __init__(self, mass, span, chord, thickness, position, angle):
            inertia_0 = 1 / 12 * mass * np.array([[thickness ** 2 + span ** 2, 0, 0],
                                                  [0, thickness ** 2 + chord ** 2, 0],
                                                  [0, 0, chord ** 2 + span ** 2]])
            self.span = span
            self.chord = chord
            self.thickness = thickness
            self.force_dcm = np.eye(3)
            self.force_mag = 0

            self.aero_table = AeroTable(aoa=np.array([0.04396,
                                                      1.12834,
                                                      1.98369,
                                                      3.10013,
                                                      4.04391,
                                                      5.16101,
                                                      6.11611,
                                                      7.07119,
                                                      8.07643,
                                                      9.03051,
                                                      9.88438,
                                                      11.10045,
                                                      12.17744,
                                                      13.06995,
                                                      14.06643,
                                                      15.11330,
                                                      16.11321,
                                                      17.06359,
                                                      18.06402,
                                                      19.06505,
                                                      20.06067,
                                                      25.00726,
                                                      30.35171,
                                                      35.35786,
                                                      39.76085,
                                                      45.09158,
                                                      50.19202,
                                                      55.22643,
                                                      59.66048,
                                                      65.04529,
                                                      69.93165,
                                                      74.96273,
                                                      79.99978,
                                                      84.53630,
                                                      89.90927,
                                                      94.94561,
                                                      99.93258,
                                                      104.92047,
                                                      109.31476,
                                                      114.76465,
                                                      119.62827,
                                                      124.63669,
                                                      129.39181,
                                                      134.81431,
                                                      139.99048,
                                                      145.21722,
                                                      149.79016,
                                                      155.10000,
                                                      160.00257,
                                                      165.16439,
                                                      169.70102,
                                                      174.97141,
                                                      180.01120,
                                                      185.52078,
                                                      190.21307,
                                                      194.89087,
                                                      200.58615,
                                                      205.36206,
                                                      209.91671,
                                                      215.07501,
                                                      219.99466,
                                                      225.40590,
                                                      230.81408,
                                                      235.36123,
                                                      240.41608,
                                                      245.24630,
                                                      250.35417,
                                                      255.51204,
                                                      260.85188,
                                                      265.01386,
                                                      270.59960,
                                                      275.01958,
                                                      280.19385,
                                                      285.30687,
                                                      290.29345,
                                                      295.26172,
                                                      300.24545,
                                                      305.26953,
                                                      310.46586,
                                                      314.80737,
                                                      320.20919,
                                                      324.90142,
                                                      330.04711,
                                                      335.24046,
                                                      339.93974,
                                                      344.92994,
                                                      350.01873,
                                                      355.03989,
                                                      360.00000]), cl=np.array([0.00000,
                                                                                0.10659,
                                                                                0.18073,
                                                                                0.25736,
                                                                                0.32399,
                                                                                0.39295,
                                                                                0.46335,
                                                                                0.53209,
                                                                                0.60040,
                                                                                0.65370,
                                                                                0.70616,
                                                                                0.76623,
                                                                                0.71510,
                                                                                0.64627,
                                                                                0.60674,
                                                                                0.57056,
                                                                                0.57147,
                                                                                0.57398,
                                                                                0.58481,
                                                                                0.58430,
                                                                                0.59255,
                                                                                0.64425,
                                                                                0.73659,
                                                                                0.80470,
                                                                                0.82920,
                                                                                0.88021,
                                                                                0.88323,
                                                                                0.83732,
                                                                                0.76842,
                                                                                0.65511,
                                                                                0.53460,
                                                                                0.40174,
                                                                                0.26268,
                                                                                0.11972,
                                                                                -0.04314,
                                                                                -0.19469,
                                                                                -0.34552,
                                                                                -0.48950,
                                                                                -0.60014,
                                                                                -0.72149,
                                                                                -0.80282,
                                                                                -0.86504,
                                                                                -0.89667,
                                                                                -0.89753,
                                                                                -0.86411,
                                                                                -0.83761,
                                                                                -0.79299,
                                                                                -0.75233,
                                                                                -0.72542,
                                                                                -0.72810,
                                                                                -0.75897,
                                                                                -0.45361,
                                                                                -0.04596,
                                                                                0.41193,
                                                                                0.62328,
                                                                                0.58515,
                                                                                0.57805,
                                                                                0.59701,
                                                                                0.63537,
                                                                                0.68047,
                                                                                0.70841,
                                                                                0.71087,
                                                                                0.69009,
                                                                                0.64279,
                                                                                0.56585,
                                                                                0.46313,
                                                                                0.29058,
                                                                                0.19244,
                                                                                0.06127,
                                                                                -0.06046,
                                                                                -0.20537,
                                                                                -0.33621,
                                                                                -0.48625,
                                                                                -0.60438,
                                                                                -0.71810,
                                                                                -0.81395,
                                                                                -0.88118,
                                                                                -0.93097,
                                                                                -0.92582,
                                                                                -0.92880,
                                                                                -0.88728,
                                                                                -0.88223,
                                                                                -0.81531,
                                                                                -0.75863,
                                                                                -0.71237,
                                                                                -0.72168,
                                                                                -0.73665,
                                                                                -0.42823,
                                                                                0.00000]), cd=np.array([0.02331,
                                                                                                        0.04492,
                                                                                                        0.03388,
                                                                                                        0.03858,
                                                                                                        0.03957,
                                                                                                        0.04251,
                                                                                                        0.04401,
                                                                                                        0.05237,
                                                                                                        0.05432,
                                                                                                        0.06903,
                                                                                                        0.07048,
                                                                                                        0.08732,
                                                                                                        0.12271,
                                                                                                        0.15381,
                                                                                                        0.19040,
                                                                                                        0.21434,
                                                                                                        0.23696,
                                                                                                        0.25710,
                                                                                                        0.27736,
                                                                                                        0.28445,
                                                                                                        0.30268,
                                                                                                        0.40028,
                                                                                                        0.54433,
                                                                                                        0.69227,
                                                                                                        0.81881,
                                                                                                        1.01019,
                                                                                                        1.19497,
                                                                                                        1.34768,
                                                                                                        1.44573,
                                                                                                        1.56713,
                                                                                                        1.64177,
                                                                                                        1.70336,
                                                                                                        1.74211,
                                                                                                        1.74923,
                                                                                                        1.75626,
                                                                                                        1.74724,
                                                                                                        1.72427,
                                                                                                        1.67187,
                                                                                                        1.60861,
                                                                                                        1.48840,
                                                                                                        1.36745,
                                                                                                        1.22983,
                                                                                                        1.07407,
                                                                                                        0.89720,
                                                                                                        0.72495,
                                                                                                        0.59485,
                                                                                                        0.48960,
                                                                                                        0.35506,
                                                                                                        0.26905,
                                                                                                        0.18551,
                                                                                                        0.10340,
                                                                                                        0.05435,
                                                                                                        0.04428,
                                                                                                        0.07934,
                                                                                                        0.14194,
                                                                                                        0.23169,
                                                                                                        0.32142,
                                                                                                        0.41407,
                                                                                                        0.52178,
                                                                                                        0.66530,
                                                                                                        0.81617,
                                                                                                        0.97441,
                                                                                                        1.14351,
                                                                                                        1.27768,
                                                                                                        1.39967,
                                                                                                        1.47868,
                                                                                                        1.55337,
                                                                                                        1.59580,
                                                                                                        1.63078,
                                                                                                        1.61361,
                                                                                                        1.59650,
                                                                                                        1.57970,
                                                                                                        1.56114,
                                                                                                        1.49730,
                                                                                                        1.43052,
                                                                                                        1.32749,
                                                                                                        1.22012,
                                                                                                        1.08527,
                                                                                                        0.95295,
                                                                                                        0.80249,
                                                                                                        0.65157,
                                                                                                        0.54600,
                                                                                                        0.42226,
                                                                                                        0.32195,
                                                                                                        0.24750,
                                                                                                        0.14903,
                                                                                                        0.03930,
                                                                                                        0.01983,
                                                                                                        0.02331]))
            super().__init__(mass, position, angle, inertia_0)

        def calculate_loads_body_frame(self, vehicle, control_manager, environment_manager, simulation_manager):
            component_velocity = self.component_velocity(vehicle)
            vel_mag = np.linalg.norm(component_velocity, ord=2)

            alpha = np.arctan2(component_velocity[2], component_velocity[0])
            lift_dir = np.cross(-component_velocity/vel_mag, np.array([0, 1, 0]))
            drag_dir = -component_velocity/vel_mag
            alpha = np.rad2deg(alpha)
            beta = np.arctan2(component_velocity[1], component_velocity[0])
            vel_eff = vel_mag * np.cos(beta)
            if alpha <= -180:
                alpha += 360
            if alpha > 180:
                alpha -= 360
            cl, cd = self.aero_table[alpha]
            rho = environment_manager.air_properties()[0]

            dyn_pres = 0.5 * rho * vel_eff ** 2 *self.chord*self.span

            drag = dyn_pres * cd * drag_dir
            lift = dyn_pres * cl * lift_dir

            if np.any(np.isnan(drag)):
                drag = np.zeros(3)
            if np.any(np.isnan(lift)):
                drag = np.zeros(3)

            force = self.body_to_component_transform.T @ (drag + lift)
            moment = np.cross(self.position - vehicle.center_of_mass, force)

            self.force_vector = force
            self.force_vector_hist.append(force)
            self.force_mag = np.linalg.norm(force,ord=2)
            self.force_ang = np.array([0, np.arctan2(force[2], np.linalg.norm(force[0:2], ord=2)),np.arctan2(force[1], force[0])])
            self.force_dcm = rotation_matrix(self.force_ang)
            return force, moment

        def calculate_angular_momentum_body_frame(self, body_rate):
            return np.zeros(3)

        def geom3d(self):
            """
                Build vertices for a colored cube.

                V  is the vertices
                I1 is the indices for a filled cube (use with GL_TRIANGLES)
                I2 is the indices for an outline cube (use with GL_LINES)
                """
            vtype = [('a_position', np.float32, 3),
                     ('a_normal', np.float32, 3),
                     ('a_color', np.float32, 4)]
            # Vertices positions
            scale = np.array([[self.chord/2, 0, 0], [0, self.span/2, 0], [0, 0, self.thickness/2]])
            v = [[1, 1, 1],
                 [-1, 1, 1],
                 [-1, -1, 1],
                 [1, -1, 1],
                 [1, -1, -1],
                 [1, 1, -1],
                 [-1, 1, -1],
                 [-1, -1, -1]]

            for ind in range(len(v)):
                v[ind] = (scale @ v[ind])#self.body_to_component_transform @

            # Face Normals
            n = [[0, 0, 1], [1, 0, 0], [0, 1, 0],
                 [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
            # Vertice colors
            c = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
                 [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]

            V = np.array([(v[0], n[0], c[0]), (v[1], n[0], c[1]),
                          (v[2], n[0], c[2]), (v[3], n[0], c[3]),
                          (v[0], n[1], c[0]), (v[3], n[1], c[3]),
                          (v[4], n[1], c[4]), (v[5], n[1], c[5]),
                          (v[0], n[2], c[0]), (v[5], n[2], c[5]),
                          (v[6], n[2], c[6]), (v[1], n[2], c[1]),
                          (v[1], n[3], c[1]), (v[6], n[3], c[6]),
                          (v[7], n[3], c[7]), (v[2], n[3], c[2]),
                          (v[7], n[4], c[7]), (v[4], n[4], c[4]),
                          (v[3], n[4], c[3]), (v[2], n[4], c[2]),
                          (v[4], n[5], c[4]), (v[7], n[5], c[7]),
                          (v[6], n[5], c[6]), (v[5], n[5], c[5])],
                         dtype=vtype)
            I1 = np.resize(np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32), 6 * (2 * 3))
            I1 += np.repeat(4 * np.arange(2 * 3, dtype=np.uint32), 6)

            I2 = np.resize(
                np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.uint32), 6 * (2 * 4))
            I2 += np.repeat(4 * np.arange(6, dtype=np.uint32), 8)

            return V, I1, I2

class RectangularPrism(Component):
    def __init__(self, mass, length, width, height, position, angle):
        inertia_0 = 1/12 * mass *np.array([[height**2 + width**2, 0, 0],
                              [0, height**2 + length**2, 0],
                              [0, 0, length**2 + width**2]])
        self.length = length
        self.width = width
        self.height = height
        super().__init__(mass, position, angle, inertia_0)

    def calculate_loads_body_frame(self, vehicle, control_manager, environment_manager, simulation_manager):
        body_velocity = vehicle.body_transform @ vehicle.velocity
        component_velocity = self.body_to_component_transform @ (body_velocity + np.cross(vehicle.ang_rate, self.position - vehicle.center_of_mass))
        vel_mag = np.linalg.norm(component_velocity, ord=2)
        volume = self.height * self.width * self.length
        cd = 0.025

        local_to_comp = self.body_to_component_transform @ vehicle.body_transform

        surface_area_yz = self.height*self.width
        surface_area_xz = self.length*self.height
        surface_area_xy = self.length*self.width

        norm_yz = local_to_comp@np.array([1,0,0])
        norm_xz = local_to_comp@np.array([0,1,0])
        norm_xy = local_to_comp@np.array([0,0,1])

        rho = environment_manager.air_properties()[0]

        drag_mod = 0.5 * cd * vel_mag**2 * rho

        drag = -drag_mod * np.abs(surface_area_yz * norm_yz + surface_area_xz * norm_xz + surface_area_xy * norm_xy) * component_velocity / vel_mag

        if np.any(np.isnan(drag)):
            drag = np.zeros(3)
        # bouyancy = vehicle.body_transform @ np.array([0,0,rho * volume * -environment_manager.gravity(simulation_manager.lla[2])[2]])

        force =  self.body_to_component_transform.T @ drag
        moment = np.cross(self.position - vehicle.center_of_mass, force)
        return  force, moment

    def calculate_angular_momentum_body_frame(self, body_rate):
            return np.zeros(3)

    def geom3d(self):
        """
            Build vertices for a colored cube.

            V  is the vertices
            I1 is the indices for a filled cube (use with GL_TRIANGLES)
            I2 is the indices for an outline cube (use with GL_LINES)
            """
        vtype = [('a_position', np.float32, 3),
                 ('a_normal', np.float32, 3),
                 ('a_color', np.float32, 4)]
        # Vertices positions
        scale = np.array([[self.length/2, 0, 0], [0, self.width/2, 0], [0, 0, self.height/2]])
        v = [[1, 1, 1],
             [-1, 1, 1],
             [-1, -1, 1],
             [1, -1, 1],
             [1, -1, -1],
             [1, 1, -1],
             [-1, 1, -1],
             [-1, -1, -1]]

        for ind in range(len(v)):
            v[ind] = self.body_to_component_transform @ (scale @ v[ind])

        # Face Normals
        n = [[0, 0, 1], [1, 0, 0], [0, 1, 0],
             [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        # Vertice colors
        c = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
             [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]

        V = np.array([(v[0], n[0], c[0]), (v[1], n[0], c[1]),
                      (v[2], n[0], c[2]), (v[3], n[0], c[3]),
                      (v[0], n[1], c[0]), (v[3], n[1], c[3]),
                      (v[4], n[1], c[4]), (v[5], n[1], c[5]),
                      (v[0], n[2], c[0]), (v[5], n[2], c[5]),
                      (v[6], n[2], c[6]), (v[1], n[2], c[1]),
                      (v[1], n[3], c[1]), (v[6], n[3], c[6]),
                      (v[7], n[3], c[7]), (v[2], n[3], c[2]),
                      (v[7], n[4], c[7]), (v[4], n[4], c[4]),
                      (v[3], n[4], c[3]), (v[2], n[4], c[2]),
                      (v[4], n[5], c[4]), (v[7], n[5], c[7]),
                      (v[6], n[5], c[6]), (v[5], n[5], c[5])],
                     dtype=vtype)
        I1 = np.resize(np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32), 6 * (2 * 3))
        I1 += np.repeat(4 * np.arange(2 * 3, dtype=np.uint32), 6)

        I2 = np.resize(
            np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.uint32), 6 * (2 * 4))
        I2 += np.repeat(4 * np.arange(6, dtype=np.uint32), 8)

        return V, I1, I2

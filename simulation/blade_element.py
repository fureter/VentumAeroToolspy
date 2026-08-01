import numpy as np

class BladeElementMethod(object):
    def __init__(self, propeller):
        self.propeller = propeller

        self.velocity_range = None
        self.rpm_range = None
        self.rho_range = None

        self.thrust_curves = None
        self.torque_curves = None


    def analyze_propeller(self, velocity_range, rpm_range, rho_range):
        mu = 18.03E-6

        self.velocity_range = velocity_range
        self.rpm_range = rpm_range
        self.rho_range = rho_range

        self.thrust_curves = np.zeros([velocity_range.shape[0], rpm_range.shape[0], rho_range.shape[0]])
        self.torque_curves = np.zeros([velocity_range.shape[0], rpm_range.shape[0], rho_range.shape[0]])

        for vel_idx in range(velocity_range.shape[0]):
            for rpm_idx in range(rpm_range.shape[0]):
                for rho_idx in range(rho_range.shape[0]):
                    vel = self.velocity_range[vel_idx]
                    rpm = self.rpm_range[rpm_idx]
                    rho = self.rho_range[rho_idx]

                    b = self.propeller.num_blades

                    # TODO: Add additional refinement below based on span resolution (i.e. dr)
                    for segment in range(self.propeller.num_segments):
                        c1 = self.propeller.segment_chords[segment]
                        c2 = self.propeller.segment_chords[segment+1]

                        b1 = self.propeller.segment_spans[segment]
                        b2 = self.propeller.segment_spans[segment+1]

                        beta1 = self.propeller.segment_twist[segment]
                        beta2 = self.propeller.segment_twist[segment+1]

                        a1 = self.propeller.airfoil_list[segment]
                        a2 = self.propeller.airfoil_list[segment + 1]

                        vel_r_1 = 2*np.pi*rpm * b1/60.0
                        vel_r_2 = 2*np.pi*rpm * b2/60.0

                        vel_1 = np.linalg.norm([vel_r_1, -vel], ord=2)
                        vel_2 = np.linalg.norm([vel_r_2, -vel], ord=2)

                        phi1 = np.arctan2(vel, vel_r_1)
                        phi2 = np.arctan2(vel, vel_r_2)

                        aoa_1 = beta1 - phi1
                        aoa_2 = beta2 - phi2

                        q1 = 0.5 * rho * vel_1**2
                        q2 = 0.5 * rho * vel_2**2

                        re1 = rho*vel_1*c1/mu
                        re2 = rho*vel_2*c2/mu

                        cl1, cd1, cm1 = a1.get_nearest_aero(re1, np.rad2deg(aoa_1))
                        cl2, cd2, cm2 = a2.get_nearest_aero(re2, np.rad2deg(aoa_2))

                        dl, dd = np.array([(q1*cl1*c1 + q2*cl2*c2)/2, (q1*cd1*c1 + q2*cd2*c2)/2])

                        self.thrust_curves[vel_idx, rpm_idx, rho_idx] += b*(dl*np.cos(phi1) - dd*np.sin(phi1))*(b2-b1)
                        self.torque_curves[vel_idx, rpm_idx, rho_idx] += b*(dl*np.sin(phi1) + dd*np.cos(phi1))*(b2-b1)*(b2+b1)/2.0


        self.propeller.valid_data = True
        self.propeller.velocity_range = velocity_range
        self.propeller.rpm_range = rpm_range
        self.propeller.rho_range = rho_range
        self.propeller.thrust_curves = self.thrust_curves
        self.propeller.torque_curves = self.torque_curves
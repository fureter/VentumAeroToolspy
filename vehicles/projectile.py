import numpy as np
from matplotlib import pyplot as plt

from simulation.simluation import SimulationManager
from .components.propulsion import IdealSolidRocket
# from torch.distributed.elastic.multiprocessing.tail_log import tail_logfile

from .components.simple_components import RectangularPrism, ThinPlate
from .vehicle import Vehicle, DefaultVehicleLoggingKeys


class SimpleDart(Vehicle):
    def __init__(self, name, tail_roll=0, tail_chord=0.3, tail_span=0.6,logger=None):
        super().__init__(name=name, logger=logger)
        self.tail_deflection = np.deg2rad(tail_roll)
        self.tail_chord = tail_chord
        self.tail_span = tail_span
        scale_mass = 5
        self.fin_deflection_max = np.deg2rad(15)

        self.fin_1 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord, thickness=0.1, position=np.array([-2, .5/2 + tail_span/2, 0.0]),
                      angle=np.array([0.0, 0.0, 0.0]))
        self.fin_2 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord, thickness=0.1, position=np.array([-2, 0.0, -(.5/2 + tail_span/2)]),
                      angle=np.array([np.deg2rad(90), 0.0, 0.0]))

        self.fin_3 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord, thickness=0.1, position=np.array([-2, -(.5/2 + tail_span/2), 0.0]),
                      angle=np.array([0.0, 0.0, 0.0]))

        self.fin_4 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord, thickness=0.1, position=np.array([-2, 0.0, .5/2 + tail_span/2]),
                      angle=np.array([np.deg2rad(90), 0.0, 0.0]))


        self.add_component(
            RectangularPrism(mass=12.5*scale_mass, length=0.5, width=0.5, height=0.5, position=np.array([0.25, 0.0, 0.0]),
                             angle=np.array([0, 0, 0])))
        self.add_component(
            RectangularPrism(mass=0.5*scale_mass, length=2, width=0.5, height=0.5, position=np.array([-1.0, 0.0, 0.0]),
                             angle=np.array([0, 0, 0])))

        self.add_component(self.fin_1)
        self.add_component(self.fin_2)
        self.add_component(self.fin_3)
        self.add_component(self.fin_4)

    def update_vehicle(self, control_manager, environment_manager, simulation_manager):
        self.control_input(control_manager)

    def control_input(self, control_manager):
        self.tail_deflection = -control_manager.roll_out

        if self.tail_deflection < -self.fin_deflection_max:
            self.tail_deflection = -self.fin_deflection_max
        if self.tail_deflection > self.fin_deflection_max:
            self.tail_deflection = self.fin_deflection_max

        self.fin_1.angle[1] = -self.tail_deflection
        self.fin_2.angle[1] = -self.tail_deflection
        self.fin_3.angle[1] = self.tail_deflection
        self.fin_4.angle[1] = self.tail_deflection

    def plot_tail_forces(self, simulation_manager):
        fin1_force=np.array(self.fin_1.force_vector_hist)
        fin2_force=np.array(self.fin_2.force_vector_hist)
        fin3_force=np.array(self.fin_3.force_vector_hist)
        fin4_force=np.array(self.fin_4.force_vector_hist)

        time = np.linspace(0, simulation_manager.time+simulation_manager.dt, fin1_force.shape[0], endpoint=False)


        plt.figure()
        plt.subplot(411)
        plt.plot(time, fin1_force[:,0])
        plt.plot(time, fin1_force[:,1])
        plt.plot(time, fin1_force[:,2])
        plt.legend(['Fx','Fy','Fz'])
        plt.title('Fin 1 Forces')
        plt.grid()
        plt.subplot(412)
        plt.plot(time, fin2_force[:,0])
        plt.plot(time, fin2_force[:,1])
        plt.plot(time, fin2_force[:,2])
        plt.legend(['Fx','Fy','Fz'])
        plt.title('Fin 2 Forces')
        plt.grid()
        plt.subplot(413)
        plt.plot(time, fin3_force[:,0])
        plt.plot(time, fin3_force[:,1])
        plt.plot(time, fin3_force[:,2])
        plt.legend(['Fx','Fy','Fz'])
        plt.title('Fin 3 Forces')
        plt.grid()
        plt.subplot(414)
        plt.plot(time, fin4_force[:,0])
        plt.plot(time, fin4_force[:,1])
        plt.plot(time, fin4_force[:,2])
        plt.legend(['Fx','Fy','Fz'])
        plt.title('Fin 4 Forces')
        plt.grid()

class SimpleWingedDart(Vehicle):
    def __init__(self, name, tail_roll=0, tail_chord=0.3, tail_span=0.3,wing_span=1.8, wing_chord=0.5,logger=None):
        super().__init__(name=name, logger=logger)
        self.tail_deflection_roll = 0
        self.tail_deflection_pitch = 0
        self.tail_deflection_yaw = 0
        self.tail_chord = tail_chord
        self.tail_span = tail_span
        self.wing_span = wing_span
        self.wing_chord = wing_chord
        scale_mass = 1
        self.fin_deflection_max = np.deg2rad(15)

        self.fin_1 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord, thickness=0.1, position=np.array([-2, .5/2 + tail_span/2, 0.0]),
                      angle=np.array([0.0, np.deg2rad(0), 0.0]))
        self.fin_2 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord, thickness=0.1, position=np.array([-2, 0.0, -(.5/2 + tail_span/2)]),
                      angle=np.array([np.deg2rad(90), 0.0, 0.0]))

        self.fin_3 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord, thickness=0.1, position=np.array([-2, -(.5/2 + tail_span/2), 0.0]),
                      angle=np.array([0.0, np.deg2rad(0), 0.0]))

        self.fin_4 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord, thickness=0.1, position=np.array([-2, 0.0, .5/2 + tail_span/2]),
                      angle=np.array([np.deg2rad(90), 0.0, 0.0]))

        self.wing_1 = ThinPlate(mass=0.1 * scale_mass, span=self.wing_span, chord=self.wing_chord, thickness=0.1,
                               position=np.array([-0.4, .5 / 2 + wing_span / 2, 0.0]),
                               angle=np.array([0.0, 0.0, 0.0]))

        self.wing_3 = ThinPlate(mass=0.1 * scale_mass, span=self.wing_span, chord=self.wing_chord, thickness=0.1,
                               position=np.array([-0.4, -(.5 / 2 + wing_span / 2), 0.0]),
                               angle=np.array([0.0, 0.0, 0.0]))

        self.add_component(
            RectangularPrism(mass=1.5*scale_mass, length=0.5, width=0.5, height=0.5, position=np.array([0.25, 0.0, 0.0]),
                             angle=np.array([0, 0, 0])))
        self.add_component(
            RectangularPrism(mass=0.5*scale_mass, length=2, width=0.5, height=0.5, position=np.array([-1.0, 0.0, 0.0]),
                             angle=np.array([0, 0, 0])))

        self.add_component(self.fin_1)
        self.add_component(self.fin_2)
        self.add_component(self.fin_3)
        self.add_component(self.fin_4)

        self.add_component(self.wing_1)
        self.add_component(self.wing_3)

    def update_vehicle(self, control_manager, environment_manager, simulation_manager):
        self.control_input(control_manager)

    def control_input(self, control_manager):
        self.tail_deflection_roll = -control_manager.roll_out
        self.tail_deflection_pitch = control_manager.pitch_out
        self.tail_deflection_yaw = -control_manager.yaw_out

        self.fin_1.angle[1] = -self.tail_deflection_roll + self.tail_deflection_pitch
        self.fin_2.angle[1] = -self.tail_deflection_roll + self.tail_deflection_yaw
        self.fin_3.angle[1] = self.tail_deflection_roll + self.tail_deflection_pitch
        self.fin_4.angle[1] = self.tail_deflection_roll + self.tail_deflection_yaw

        if abs(self.fin_1.angle[1]) > self.fin_deflection_max:
            self.fin_1.angle[1] = self.fin_deflection_max * self.fin_1.angle[1]/abs(self.fin_1.angle[1])
        if abs(self.fin_2.angle[1]) > self.fin_deflection_max:
            self.fin_2.angle[1] = self.fin_deflection_max * self.fin_2.angle[1]/abs(self.fin_2.angle[1])
        if abs(self.fin_3.angle[1]) > self.fin_deflection_max:
            self.fin_3.angle[1] = self.fin_deflection_max * self.fin_3.angle[1]/abs(self.fin_3.angle[1])
        if abs(self.fin_4.angle[1]) > self.fin_deflection_max:
            self.fin_4.angle[1] = self.fin_deflection_max * self.fin_4.angle[1]/abs(self.fin_4.angle[1])

    def plot_tail_forces(self, simulation_manager):
        fin1_force=np.array(self.fin_1.force_vector_hist)
        fin2_force=np.array(self.fin_2.force_vector_hist)
        fin3_force=np.array(self.fin_3.force_vector_hist)
        fin4_force=np.array(self.fin_4.force_vector_hist)

        time = np.linspace(0, simulation_manager.time+simulation_manager.dt, fin1_force.shape[0], endpoint=False)


        plt.figure()
        plt.subplot(411)
        plt.plot(time, fin1_force[:,0])
        plt.plot(time, fin1_force[:,1])
        plt.plot(time, fin1_force[:,2])
        plt.legend(['Fx','Fy','Fz'])
        plt.title('Fin 1 Forces')
        plt.grid()
        plt.subplot(412)
        plt.plot(time, fin2_force[:,0])
        plt.plot(time, fin2_force[:,1])
        plt.plot(time, fin2_force[:,2])
        plt.legend(['Fx','Fy','Fz'])
        plt.title('Fin 2 Forces')
        plt.grid()
        plt.subplot(413)
        plt.plot(time, fin3_force[:,0])
        plt.plot(time, fin3_force[:,1])
        plt.plot(time, fin3_force[:,2])
        plt.legend(['Fx','Fy','Fz'])
        plt.title('Fin 3 Forces')
        plt.grid()
        plt.subplot(414)
        plt.plot(time, fin4_force[:,0])
        plt.plot(time, fin4_force[:,1])
        plt.plot(time, fin4_force[:,2])
        plt.legend(['Fx','Fy','Fz'])
        plt.title('Fin 4 Forces')
        plt.grid()


class DeployableWingedDart(Vehicle):
    class DataKeys(object):
        FIN1_DEFLECTION = 'Fin1 Deflection'
        FIN2_DEFLECTION = 'Fin2 Deflection'
        FIN3_DEFLECTION = 'Fin3 Deflection'
        FIN4_DEFLECTION = 'Fin4 Deflection'
        ALL_FINS = [FIN1_DEFLECTION, FIN2_DEFLECTION, FIN3_DEFLECTION, FIN4_DEFLECTION]

        ROLL_OUT = 'Roll Out'
        PITCH_OUT = 'Pitch Out'
        YAW_OUT = 'Yaw Out'
        ALL_OUTPUT = [ROLL_OUT, PITCH_OUT, YAW_OUT]

        ROLL_CONTROL = [ROLL_OUT, SimulationManager.DefaultSimulationLoggingKeys.ROLL]
        PITCH_CONTROL = [PITCH_OUT, SimulationManager.DefaultSimulationLoggingKeys.PITCH]
        YAW_CONTROL = [YAW_OUT, SimulationManager.DefaultSimulationLoggingKeys.YAW]

    def __init__(self, name, tail_chord=0.25, tail_span=0.5, wing_span=2.0, wing_chord=0.18,data_logger=None):
        super().__init__(name=name, data_logger=data_logger)
        self.tail_deflection_roll = 0
        self.tail_deflection_pitch = 0
        self.tail_deflection_yaw = 0
        self.tail_chord = tail_chord
        self.tail_span = tail_span
        self.wing_span = wing_span
        self.wing_chord = wing_chord

        self.wing_anchor_point = np.array([-0.1, 0.0, 0.0])
        self.wing_sweep_angle = np.deg2rad(90)
        self.wing_sweep_rate = 2.0 #rad/sec
        self.wing_aoi = np.deg2rad(4.0)
        self.control_active = False
        self.deploy_wing = False
        scale_mass = 4
        self.fin_deflection_max = np.deg2rad(30)

        self.fin_1 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord, thickness=0.1, position=np.array([-2, .5/2 + tail_span/2, 0.0]),
                      angle=np.array([0.0, np.deg2rad(0), 0.0]))
        self.fin_2 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord*1.5, thickness=0.1, position=np.array([-2, 0.0, -(.5/2 + tail_span/2)]),
                      angle=np.array([np.deg2rad(90), 0.0, 0.0]))

        self.fin_3 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord, thickness=0.1, position=np.array([-2, -(.5/2 + tail_span/2), 0.0]),
                      angle=np.array([0.0, np.deg2rad(0), 0.0]))

        self.fin_4 = ThinPlate(mass=0.1*scale_mass, span=self.tail_span, chord=self.tail_chord*1.5, thickness=0.1, position=np.array([-2, 0.0, .5/2 + tail_span/2]),
                      angle=np.array([np.deg2rad(90), 0.0, 0.0]))

        self.wing_1 = ThinPlate(mass=0.5 * scale_mass, span=self.wing_span, chord=self.wing_chord, thickness=0.1,
                               position=np.array([self.wing_anchor_point[0] - np.sin(self.wing_sweep_angle) * self.wing_span/2, self.wing_anchor_point[1] + np.cos(self.wing_sweep_angle) * self.wing_span/2, 0.0]),
                               angle=np.array([0.0, self.wing_aoi, -self.wing_sweep_angle]))

        self.wing_3 = ThinPlate(mass=0.5 * scale_mass, span=self.wing_span, chord=self.wing_chord, thickness=0.1,
                               position=np.array([self.wing_anchor_point[0] - np.sin(self.wing_sweep_angle) * self.wing_span/2, self.wing_anchor_point[1] - np.cos(self.wing_sweep_angle) * self.wing_span/2, 0.0]),
                               angle=np.array([0.0, self.wing_aoi, self.wing_sweep_angle]))

        self.rocket = IdealSolidRocket(mass=8.50, position=np.array([-2.0, 0.0, 0.0]), angle=np.array([0, np.deg2rad(0), 0]),
                                       specific_isp=180.0, average_thrust=150.0, fuel_mass_fraction=0.98)
        self.add_component(self.rocket)

        self.add_component(
            RectangularPrism(mass=1.0*scale_mass, length=0.5, width=0.5, height=0.5, position=np.array([0.25, 0.0, 0.0]),
                             angle=np.array([0, 0, 0])))
        self.add_component(
            RectangularPrism(mass=0.5*scale_mass, length=2, width=0.5, height=0.5, position=np.array([-1.0, 0.0, 0.0]),
                             angle=np.array([0, 0, 0])))

        self.add_component(self.fin_1)
        self.add_component(self.fin_2)
        self.add_component(self.fin_3)
        self.add_component(self.fin_4)

        self.add_component(self.wing_1)
        self.add_component(self.wing_3)

    def calculate_loads_body_frame(self, control_manager, environment_manager, simulation_manager):
        force = np.zeros(3)
        moment = np.zeros(3)

        for component in self.components:
            f, m = component.calculate_loads_body_frame(self, control_manager, environment_manager, simulation_manager)
            if self.rocket.burn_complete:
                continue
            force += f
            moment += m
        self.force = force
        self.moment = moment

        return force, moment

    def calculate_angular_momentum_body_frame(self):
        angular_momentum = np.array(np.zeros(3))
        for component in self.components:
            angular_momentum += component.calculate_angular_momentum_body_frame(self.ang_rate)

        return angular_momentum

    def move_origin_to_center_of_mass(self):
        self.wing_anchor_point -= self.center_of_mass
        super().move_origin_to_center_of_mass()


    def update_vehicle(self, control_manager, environment_manager, simulation_manager):
        if np.rad2deg(self.pitch) < 40:
            control_manager.active=True
        self.control_input(control_manager)

        self.rocket.process_fuel_consumption(self, control_manager, environment_manager, simulation_manager)
        self.rocket.burn_start = True

        if self.deploy_wing:
            control_manager.pitch_target = np.deg2rad(45) if not self.rocket.burn_complete else np.deg2rad(5.0)
            if (self.wing_sweep_angle > 0 and self.rocket.burn_complete) or (self.wing_sweep_angle > np.deg2rad(20)):
                self.wing_sweep_angle -= self.wing_sweep_rate*simulation_manager.dt
            else:
                self.wing_sweep_angle = 0 if self.rocket.burn_complete else np.deg2rad(20)

            self.wing_1.position = np.array([self.wing_anchor_point[0] - np.sin(self.wing_sweep_angle) * self.wing_span/2, self.wing_anchor_point[1] + np.cos(self.wing_sweep_angle) * self.wing_span/2, self.wing_1.position[2]])
            self.wing_3.position = np.array([self.wing_anchor_point[0] - np.sin(self.wing_sweep_angle) * self.wing_span/2, self.wing_anchor_point[1] - np.cos(self.wing_sweep_angle) * self.wing_span/2, self.wing_3.position[2]])
            self.wing_1.angle = np.array([0.0, self.wing_aoi, -self.wing_sweep_angle])
            self.wing_3.angle = np.array([0.0, self.wing_aoi, self.wing_sweep_angle])

            self.inertia_changed = True

    def log_iteration(self, time):
        super().log_iteration(time)

        self.data_logger.add_data_items(time, items=self.DataKeys.ALL_FINS, datas=np.rad2deg([self.fin_1.angle[1],self.fin_2.angle[1],self.fin_3.angle[1],self.fin_4.angle[1]]))
        self.data_logger.add_data_items(time, items=self.DataKeys.ALL_OUTPUT, datas=[self.tail_deflection_roll,self.tail_deflection_pitch,self.tail_deflection_yaw])

    def control_input(self, control_manager):
        self.tail_deflection_roll = control_manager.roll_out / 80
        self.tail_deflection_pitch = -control_manager.pitch_out / 20.0
        self.tail_deflection_yaw = control_manager.yaw_out / 20.0

        if control_manager.roll_out != 0:
            self.deploy_wing = True

        self.fin_1.angle[1] = (-self.tail_deflection_roll + self.tail_deflection_pitch)/2
        self.fin_2.angle[1] = (-self.tail_deflection_roll + self.tail_deflection_yaw)/2
        self.fin_3.angle[1] = (self.tail_deflection_roll + self.tail_deflection_pitch)/2
        self.fin_4.angle[1] = (self.tail_deflection_roll + self.tail_deflection_yaw)/2

        if abs(self.fin_1.angle[1]) > self.fin_deflection_max:
            self.fin_1.angle[1] = self.fin_deflection_max * self.fin_1.angle[1]/abs(self.fin_1.angle[1])
        if abs(self.fin_2.angle[1]) > self.fin_deflection_max:
            self.fin_2.angle[1] = self.fin_deflection_max * self.fin_2.angle[1]/abs(self.fin_2.angle[1])
        if abs(self.fin_3.angle[1]) > self.fin_deflection_max:
            self.fin_3.angle[1] = self.fin_deflection_max * self.fin_3.angle[1]/abs(self.fin_3.angle[1])
        if abs(self.fin_4.angle[1]) > self.fin_deflection_max:
            self.fin_4.angle[1] = self.fin_deflection_max * self.fin_4.angle[1]/abs(self.fin_4.angle[1])

    def log_data(self, time, data_logger):
        raise NotImplementedError('Not implemented for vehicle type')

    def plot_tail_forces(self):
        self.data_logger.simple_plot(self.DataKeys.ALL_FINS, time_frame=None,
                                     subplot=False)
        self.data_logger.simple_plot(self.DataKeys.ALL_OUTPUT, time_frame=None,
                                     subplot=False)
        self.data_logger.simple_plot(self.DataKeys.YAW_CONTROL, time_frame=None,
                                     subplot=False)
        self.data_logger.simple_plot(self.DataKeys.ROLL_CONTROL, time_frame=None,
                                     subplot=False)
        self.data_logger.simple_plot(self.DataKeys.PITCH_CONTROL, time_frame=None,
                                     subplot=False)


import numpy as np
from matplotlib import pyplot as plt
from torch.distributed.elastic.multiprocessing.tail_log import tail_logfile

from .components.simple_components import RectangularPrism, ThinPlate
from .vehicle import Vehicle

class SimpleDart(Vehicle):
    def __init__(self, name, tail_roll=0, tail_chord=0.3, tail_span=0.6,logger=None):
        super().__init__(name=name, logger=logger)
        self.tail_deflection = np.deg2rad(tail_roll)
        self.tail_chord = tail_chord
        self.tail_span = tail_span
        scale_mass = 5
        self.fin_deflection_max = np.deg2rad(15)

        # self.add_component(
        #     ThinPlate(mass=2*scale_mass, span=self.tail_span*2, chord=self.tail_chord, thickness=0.1, position=np.array([2.5, 0.0, 0.0]),
        #               angle=np.array([0, 0, 0])))
        #
        # self.add_component(
        #     ThinPlate(mass=1*scale_mass, span=self.tail_span*15, chord=self.tail_chord*1.2, thickness=0.1, position=np.array([0.0, 0.0, 0.0]),
        #               angle=np.array([0, 0, 0])))
        # offset = self.tail_span*15/2 * np.sin(np.deg2rad(5))
        # self.add_component(
        #     ThinPlate(mass=1*scale_mass, span=self.tail_span*15, chord=self.tail_chord*1.2, thickness=0.1, position=np.array([0.0, -self.tail_span*15, offset]),
        #               angle=np.array([np.deg2rad(5), 0, 0])))
        # self.add_component(
        #     ThinPlate(mass=1*scale_mass, span=self.tail_span*15, chord=self.tail_chord*1.2, thickness=0.1, position=np.array([0.0, self.tail_span*15, offset]),
        #               angle=np.array([-np.deg2rad(5), 0, 0])))
        #
        # self.add_component(
        #     ThinPlate(mass=0.7 * scale_mass, span=self.tail_span * 10, chord=self.tail_chord*1.5, thickness=0.1,
        #               position=np.array([-4.8, 0.0, 0.3]),
        #               angle=np.array([0, self.tail_roll*2, 0])))
        # self.add_component(
        #     ThinPlate(mass=0.5 * scale_mass, span=self.tail_span * 6, chord=self.tail_chord*2, thickness=0.1,
        #               position=np.array([-4.5, 0.0, 0.3 + self.tail_span * 6/2]),
        #               angle=np.array([np.deg2rad(90), 0, 0])))
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


    def calculate_loads_body_frame(self, control_manager, environment_manager, simulation_manager):
        force = np.zeros(3)
        moment = np.zeros(3)

        for component in self.components:
            f, m = component.calculate_loads_body_frame(self, control_manager, environment_manager, simulation_manager)
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
    def __init__(self, name, tail_roll=0, tail_chord=0.3, tail_span=0.3,wing_span=1.2, wing_chord=0.5,logger=None):
        super().__init__(name=name, logger=logger)
        self.tail_deflection_roll = 0
        self.tail_deflection_pitch = 0
        self.tail_deflection_yaw = 0
        self.tail_chord = tail_chord
        self.tail_span = tail_span
        self.wing_span = wing_span
        self.wing_chord = wing_chord
        scale_mass = 5
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
                               position=np.array([0.15, .5 / 2 + wing_span / 2, 0.0]),
                               angle=np.array([0.0, 0.0, 0.0]))
        # self.wing_2 = ThinPlate(mass=0.1 * scale_mass, span=self.wing_span, chord=self.wing_chord, thickness=0.1,
        #                        position=np.array([0.15, 0.0, -(.5 / 2 + wing_span / 2)]),
        #                        angle=np.array([np.deg2rad(90), 0.0, 0.0]))

        self.wing_3 = ThinPlate(mass=0.1 * scale_mass, span=self.wing_span, chord=self.wing_chord, thickness=0.1,
                               position=np.array([0.15, -(.5 / 2 + wing_span / 2), 0.0]),
                               angle=np.array([0.0, 0.0, 0.0]))

        # self.wing_4 = ThinPlate(mass=0.1 * scale_mass, span=self.wing_span, chord=self.wing_chord, thickness=0.1,
        #                        position=np.array([0.15, 0.0, .5 / 2 + wing_span / 2]),
        #                        angle=np.array([np.deg2rad(90), 0.0, 0.0]))

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

        self.add_component(self.wing_1)
        # self.add_component(self.wing_2)
        self.add_component(self.wing_3)
        # self.add_component(self.wing_4)

    def calculate_loads_body_frame(self, control_manager, environment_manager, simulation_manager):
        force = np.zeros(3)
        moment = np.zeros(3)

        for component in self.components:
            f, m = component.calculate_loads_body_frame(self, control_manager, environment_manager, simulation_manager)
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

    def control_input(self, control_manager):
        self.tail_deflection_roll = -control_manager.roll_out
        self.tail_deflection_pitch = control_manager.pitch_out
        self.tail_deflection_yaw = control_manager.yaw_out

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


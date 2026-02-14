import copy

import numpy as np
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate

from vehicles import vehicle
from vehicles.components.simple_components import RectangularPrism
from visualize.pimative_components import frag, vert, Arrow


def plot_vehicle(vehicle, simulation_manager, sim_rate, sim_callback):
    canvas = Canvas(vehicle, simulation_manager, sim_rate, sim_callback)
    app.run()

def to_camera_frame(position):
    tmp = copy.deepcopy(position)
    position[3] = tmp[2]
    position[2] = tmp[3]
    return position

# -----------------------------------------------------------------------------
class Canvas(app.Canvas):

    def __init__(self, vehicle, simulation_manager, sim_rate, sim_callback):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))

        self.vehicle = vehicle
        self.simulation_manager = simulation_manager
        self.program = gloo.Program(vert, frag)
        self.vertices_buffers = dict()
        self.filled_buffer = dict()
        self.outline_buffer = dict()

        for component in vehicle.components:
            vertices, filled, outline = component.geom3d()
            self.vertices_buffers[component.num_id] = gloo.VertexBuffer(vertices)
            self.filled_buffer[component.num_id] = gloo.IndexBuffer(filled)
            self.outline_buffer[component.num_id] = gloo.IndexBuffer(outline)

        vertices, filled, outline = Arrow()
        self.vertices_buffers[-1] = gloo.VertexBuffer(vertices)
        self.filled_buffer[-1] = gloo.IndexBuffer(filled)
        self.outline_buffer[-1] = gloo.IndexBuffer(outline)

        self.view = translate((0, 0, -5))

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 2.0, 10.0)

        self.program['u_projection'] = self.projection
        self.program['u_view'] = self.view


        gloo.set_clear_color('white')
        gloo.set_state('opaque')
        gloo.set_polygon_offset(1, 1)

        self._timer = app.Timer(1/sim_rate, connect=self.on_timer, start=True)
        self.sim_callback = sim_callback
        self.time = 0

        self.show()

    # ---------------------------------
    def on_timer(self, event):
        self.sim_callback(1)
        self.update()

    # ---------------------------------
    def on_resize(self, event):
        gloo.set_viewport(0, 0, event.physical_size[0], event.physical_size[1])
        self.projection = perspective(80.0, event.size[0] /
                                      float(event.size[1]), 2.0, 50.0)
        self.program['u_projection'] = self.projection

    # ---------------------------------
    def on_draw(self, event):
        gloo.clear()

        veh_pos = self.vehicle.position
        self.view = translate((-veh_pos[0], -veh_pos[1], -veh_pos[2]-20))
        self.program['u_view'] = self.view

        for ind in range(self.vehicle.num_components):
            comp_id = self.vehicle.components[ind].num_id
            force = self.vehicle.components[ind].force_vector
            dcm =  (self.vehicle.body_transform @ self.vehicle.components[ind].body_to_component_transform ).T
            position = np.zeros(4)
            position[:3] = veh_pos + self.vehicle.body_transform @ self.vehicle.components[ind].position
            # position = to_camera_frame(position)

            model = np.eye(4, dtype=np.float32)
            model[:3, :3] = dcm
            self.program['u_model'] = model
            self.program['u_position'] = position

            self.program.bind(self.vertices_buffers[comp_id])
            gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
            self.program['u_color'] = 1, 1, 1, 1
            self.program.draw('triangles', self.filled_buffer[comp_id])

            # Outline
            gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=False)
            gloo.set_depth_mask(False)
            self.program['u_color'] = 0, 0, 0, 1
            self.program.draw('lines', self.outline_buffer[comp_id])
            gloo.set_depth_mask(True)

            # self.program.bind(self.vertices_buffers[-1])
            # force_vec_dcm = self.vehicle.components[ind].force_dcm @ dcm
            # model[:3, :3] = force_vec_dcm
            # model[2,2] *= self.vehicle.components[ind].force_mag
            # self.program['u_model'] = model
            # gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
            # self.program['u_color'] = 1, 1, 1, 1
            # self.program.draw('triangles', self.filled_buffer[-1])
            # gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=False)
            # gloo.set_depth_mask(False)
            # self.program['u_color'] = 0, 0, 0, 1
            # self.program.draw('lines', self.outline_buffer[-1])
            # gloo.set_depth_mask(True)

import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

def simulate_2d_static_emitter_field(emitters, bounding_box, spatial_resolution, speed_of_sound=337, phase_offsets=None,
                                     output_dir=None, title=None):
    """
    
    :param emitters: 
    :param bounding_box: 
    :param spatial_resolution: 
    :return: 
    """
    if phase_offsets is None:
        phase_offsets = np.zeros(len(emitters))
    nu =1.8E-5
    rho = 1.225
    num_x = int(((bounding_box[0,1] - bounding_box[0,0]) / spatial_resolution))+1
    num_y = int(((bounding_box[1,1] - bounding_box[1,0]) / spatial_resolution))+1
    amplitude_field = np.zeros([num_x, num_y])

    x = np.linspace(bounding_box[0,0], bounding_box[0,1], num_x, endpoint=True)
    y = np.linspace(bounding_box[1,0], bounding_box[1,1], num_y, endpoint=True)

    x_mesh, y_mesh = np.meshgrid(x, y)
    plt.figure(figsize=(16,9), dpi=72)
    for ind, emitter in enumerate(emitters):
        alpha = 2 * nu * (2*np.pi*emitter.base_frequency)**2/(3*rho*speed_of_sound**3)
        distance = np.sqrt((x_mesh-emitter.position[0])**2 + (y_mesh-emitter.position[1])**2)
        # distance = np.sqrt(x_mesh**2 + y_mesh**2)
        time = distance/speed_of_sound + phase_offsets[ind]/(2*np.pi*emitter.base_frequency)
        power = 10**(emitter.emit_power/10) * np.exp(-alpha * distance)
        omega_t = 2*np.pi*emitter.base_frequency * time
        wave_length = speed_of_sound / emitter.base_frequency
        wave_number = 2*np.pi/wave_length
        k_x = distance * wave_number
        amplitude_field += np.cos(omega_t.T)*power.T
    # amplitude_field = 10*np.log10(amplitude_field)

    plt.contourf(x, y, amplitude_field.T,  cmap=mpl.colormaps['seismic'], levels=64)
    plt.colorbar()
    for emitter in emitters:
        plt.scatter(emitter.position[0], emitter.position[1])
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, '%s.png' % title))
    else:
        plt.show()




import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

# Based mostly on Van Trees Optimum Array Processing
class AntennaPattern(object):
    def __init__(self, e_rhcp, e_lhcp, v_pol, h_pol, az, el, frequency_range):
        self.e_rhcp = e_rhcp
        self.e_lhcp = e_lhcp
        self.v_pol = v_pol
        self.h_pol = h_pol
        self.az = az
        self.el = el
        self.freq = frequency_range
        self.num_elem = e_rhcp.shape[3] if e_rhcp is not None else (e_lhcp.shape[3] if e_lhcp is not None else (v_pol.shape[3] if v_pol is not None else h_pol.shape[3]))

    def get_antenna_response(self, az, el, polarization='RHCP'):
        az_ind = abs(az - self.az).argmin()
        el_ind = abs(el - self.el).argmin()
        return self.e_rhcp[az_ind, el_ind, :, :]

    @classmethod
    def gen_ideal_omni_array(cls, elem_position, frequency_range, az_resolution, el_resolution, elem_clocking=None,
                             wavespeed=constants.c):
        az = np.linspace(-np.pi, np.pi, int(2*np.pi/az_resolution))
        el = np.linspace(0.0, np.pi, int(np.pi/el_resolution))
        num_elem = elem_position.shape[0]
        if elem_clocking is None:
            elem_clocking = np.zeros(num_elem, dtype=np.complex128)
        e_rhcp = np.zeros([az.shape[0], el.shape[0], frequency_range.shape[0], num_elem], dtype=np.complex128)
        for az_ind, azimuth in enumerate(az):
            ca = np.cos(azimuth)
            sa = np.sin(azimuth)
            for el_ind, elevation in enumerate(el):
                ce = np.cos(elevation)
                se = np.sin(elevation)
                for freq_ind, freq in enumerate(frequency_range):
                    lam = wavespeed / freq
                    k = 2 * np.pi / lam
                    for elem in range(num_elem):
                        e_rhcp[az_ind, el_ind, freq_ind, elem] = np.exp(1j*(elem_clocking[elem] + k*np.array([[se * ca],
                                                                                          [se * sa],
                                                                                          [ce]]).T @ elem_position[elem]))

        return AntennaPattern(e_rhcp, None, None, None, az, el, frequency_range)


    @classmethod
    def from_elements(cls, elements):
        pass

    def plot_azimuth_slice(self, output_dir=None, show=False, title=None):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': 'polar'},
                                layout='constrained')
        theta = self.el
        forward_az_index = abs(self.az-0).argmin()
        backward_az_index = abs(self.az - np.pi).argmin()
        ax.set_rlim(bottom = -30, top=10)
        ax.set_theta_zero_location('N')
        for freq in range(self.freq.shape[0]):
            for elem in range(self.num_elem):
                ax.plot(theta, 10*np.log10(self.e_rhcp[forward_az_index, :, freq, elem]), color='C%s' % (freq*self.num_elem + elem))
                ax.plot(-theta, 10*np.log10(np.abs(self.e_rhcp[backward_az_index, :, freq, elem])), color='C%s'% (freq*self.num_elem + elem))
        plt.legend()
        if title is not None:
            ax.set_title(title)
        if show:
            plt.show()
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, title + '.png'))

    def plot_gain_hemispheres(self, freq, output_dir=None, show=False, title=None):
        fig, axs = plt.subplots(2, self.num_elem, figsize=(8, 8), subplot_kw={'projection': 'polar'},
                                layout='constrained')
        freq_index = abs(freq - self.freq).argmin()
        az_mesh, el_mesh = np.meshgrid(self.az, self.el)
        for ind in [0,1]:
            for elem in range(self.num_elem):
                ax = axs[ind, elem]
                ax.set_rlim(bottom = 90 if ind == 0 else -90, top=0 if ind == 0 else 0)
                ax.pcolormesh(az_mesh, 90-np.rad2deg(el_mesh), 10*np.log10(abs(self.e_rhcp[:, :, freq_index, elem])))
        if title is not None:
            plt.set_title(title)
        if show:
            plt.show()
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, title + '.png'))

    def plot_phase_hemispheres(self, freq, output_dir=None, show=False, title=None):
        fig, axs = plt.subplots(2, self.num_elem, figsize=(8, 8), subplot_kw={'projection': 'polar'},
                                layout='constrained')
        freq_index = abs(freq - self.freq).argmin()
        az_mesh, el_mesh = np.meshgrid(self.az, np.rad2deg(self.el))
        for ind in [0,1]:
            for elem in range(self.num_elem):
                ax = axs[ind, elem]
                ax.set_rlim(bottom = 90 if ind == 0 else -90, top=0 if ind == 0 else 0)
                ax.pcolormesh(az_mesh, 90-el_mesh, np.angle(self.e_rhcp[:, :, freq_index, elem]))
        if title is not None:
            plt.set_title(title)
        if show:
            plt.show()
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, title + '.png'))


class AntennaFactory(object):
    pass


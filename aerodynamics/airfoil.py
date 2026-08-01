import aerosandbox as asb
import aerosandbox.numpy as np

class Airfoil(object):

    def __init__(self, name):
        self.name = name
        self.re_list = None
        self.aoa_list = None

        self.aero_data = None


    def get_nearest_aero(self, re, aoa):
        re_idx = np.argmin(np.abs(re - self.re_list))
        aoa_idx = np.argmin(np.abs(aoa - self.aoa_list))
        return self.aero_data[re_idx, aoa_idx, :]


    @staticmethod
    def load_from_folder_xfoil(folder_path):
        pass

class NerualAirfoil(Airfoil):
    def __init__(self, name, re_list, aoa_list):
        super(NerualAirfoil, self).__init__(name)
        self.re_list = re_list
        self.aoa_list = aoa_list

        self.af = asb.Airfoil(name)
        self.aero = self.af.get_aero_from_neuralfoil(
            alpha=self.aoa_list,
            Re=self.re_list,
            mach=0,
        )

        self.aero_data = np.zeros([len(self.re_list), len(self.aoa_list), 3])
        self.aero_data[:, :, 0] = self.aero['CL']
        self.aero_data[:, :, 1] = self.aero['CD']
        self.aero_data[:, :, 2] = self.aero['CM']

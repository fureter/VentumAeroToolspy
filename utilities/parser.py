import os

import numpy as np
import matplotlib.pyplot as plt


class Parser():
    """Base Parser class to be extended by implemented parser classes for specific file types.

    :param self._data: Array of Point entries representing points along the curve defined by the dat file.
    :type self._data: np.ndarray.
    :param self._filepath: path to .dat file either for import or export.
    :type self._filepath: str.
    """

    def __init__(self, data=None, filepath=None):
        self.name = None
        if data is not None:
            self._data = data
        if filepath is not None:
            self._filepath = filepath
            self._import_data()

    def get_data(self):
        """Return parsed data."""
        return self._data

    def _import_data(self):
        """Abstract function for importing data, implemented by inherited classes."""
        raise NotImplementedError()

    def export_data(self, filepath):
        """Abstract function for exporting data to a file, implemented by inherited classes."""
        raise NotImplementedError()

    def log(self, logger):
        """Log the data from the parser using a logger instance"""
        logger.info('Printing Data %s\r\n' % self.name)
        for data in self._data:
            logger.info(str(data))


class Dat(Parser):
    """Parser implementation for .dat files. Can import 2D and 3D dat files, irrespective of point order.
    extension of :class: `Parser`

    :param self._data: Array of Point entries representing points along the curve defined by the dat file.
    :type self._data: np.ndarray.
    :param self._filepath: path to .dat file either for import or export.
    :type self._filepath: str.
    :param self._type: Defines the whether the data is from a 2D or a 3D source.
    :type self._type: str.
    """

    def __init__(self, data=None, filepath=None):
        """Constructor for Dat parser. Accepts either a list of data, or a filepath to import data from.

        :param data: Array of Point entries representing points along the curve defined by the dat file.
        :type data: np.ndarray.
        :param filepath: path to .dat file either for import or export.
        :type filepath: str.
        """
        super().__init__(data, filepath)
        if filepath is None:
            self.type = None
            self.name = None
        else:
            self._import_data()

    def _import_data(self):
        """Implementation of Parser _import_data function. Reads through each line of the .dat file and creates 3D
        points from the data.
        """
        if self._filepath is None:
            raise AttributeError('Filepath is missing for data import')

        with open(self._filepath) as file:
            lines = file.readlines()
            self._data = np.zeros([len(lines), 3])
            for line_num, row in enumerate(lines):
                split = row.split(' ')  # most dat files use spaces to separate the values instead of commas.

                if len(split) == 1:  # if there were no spaces, the file is most likely using \t characters
                    split = split[0].split('\t')  # re-split with \t

                for indx in range(0, len(split)):
                    # check if the value is cast-able as a float (some dat files have headers we want to ignore)
                    if is_float(split[indx]):
                        self._data[line_num, indx] = float(split[indx])
                    # check if the split has a new line character, if so remove it
                    if r'\n' in split[indx]:
                        self._data[line_num, indx] = float(split[indx][:-2])
        if self.name is None:
            self.name = os.path.basename(self._filepath).rstrip()

    def export_data(self, filepath=None):
        """Export data implementation for Dat files.

        :param filepath: Filepath to the file to save the new dat file as. If not provided will use self._filepath.
        """
        # currently not implemented, call super function to raise error
        super().export_data(filepath)

    def plot_points_2d(self):
        """plot the x,y coordinates of the dat file to visualize the file. Assumes the x and y dimensions are the ones
        containing the relevant data, z is assumed empty (not always the case). Function does not show plot, relies on
        higher level implementation to display the plot so that multiple plots can be stacked up.
        """
        size = len(self._data)
        x = np.zeros(size)
        y = np.zeros(size)

        plt.plot(self._data[:,0], self._data[:,1], 'vr')
        plt.plot(self._data[:,0], self._data[:,1], 'k')

class Xfoil(Parser):
    """Parser implementation for .dat files. Can import 2D and 3D dat files, irrespective of point order.
    extension of :class: `Parser`

    :param self._data: Array of Point entries representing points along the curve defined by the dat file.
    :type self._data: np.ndarray.
    :param self._filepath: path to .dat file either for import or export.
    :type self._filepath: str.
    :param self._type: Defines the whether the data is from a 2D or a 3D source.
    :type self._type: str.
    """

    def __init__(self, data=None, filepath=None):
        """Constructor for Dat parser. Accepts either a list of data, or a filepath to import data from.

        :param data: Array of Point entries representing points along the curve defined by the dat file.
        :type data: np.ndarray.
        :param filepath: path to .dat file either for import or export.
        :type filepath: str.
        """
        super().__init__(data, filepath)
        if filepath is None:
            self.type = None
            self.name = None
        else:
            self._import_data()

    def _import_data(self):
        """Implementation of Parser _import_data function. Reads through each line of the .dat file and creates 3D
        points from the data.
        """
        if self._filepath is None:
            raise AttributeError('Filepath is missing for data import')

        with open(self._filepath) as file:
            lines = file.readlines()
            self._data = np.zeros([len(lines), 3])
            for line_num, row in enumerate(lines):
                split = row.split(' ')  # most dat files use spaces to separate the values instead of commas.

                if len(split) == 1:  # if there were no spaces, the file is most likely using \t characters
                    split = split[0].split('\t')  # re-split with \t

                for indx in range(0, len(split)):
                    # check if the value is cast-able as a float (some dat files have headers we want to ignore)
                    if is_float(split[indx]):
                        self._data[line_num, indx] = float(split[indx])
                    # check if the split has a new line character, if so remove it
                    if r'\n' in split[indx]:
                        self._data[line_num, indx] = float(split[indx][:-2])
        if self.name is None:
            self.name = os.path.basename(self._filepath).rstrip()

    def export_data(self, filepath=None):
        """Export data implementation for Dat files.

        :param filepath: Filepath to the file to save the new dat file as. If not provided will use self._filepath.
        """
        # currently not implemented, call super function to raise error
        super().export_data(filepath)

    def plot_points_2d(self):
        """plot the x,y coordinates of the dat file to visualize the file. Assumes the x and y dimensions are the ones
        containing the relevant data, z is assumed empty (not always the case). Function does not show plot, relies on
        higher level implementation to display the plot so that multiple plots can be stacked up.
        """
        size = len(self._data)
        x = np.zeros(size)
        y = np.zeros(size)

        plt.plot(self._data[:,0], self._data[:,1], 'vr')
        plt.plot(self._data[:,0], self._data[:,1], 'k')


def is_float(val):
    """Check whether val is cast-able as a float.

    :param val: Value to check if cast-able as a float.
    :type val: unknown
    :return: True or False as to whether val can be cas-table as a float.
    """
    ret_val = False
    try:
        float(val)
        ret_val = True
    finally:
        return ret_val

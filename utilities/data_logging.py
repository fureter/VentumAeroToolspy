import numpy as np
import matplotlib.pyplot as plt

class DataLogger(object):
    """
    Data logger tracks arbitrary data throughout VentumAeroTools. Data is stored in
    """

    def __init__(self):
        self.data = dict()
        self.items = list()

    def register_item(self, item):
        self.items.append(item)
        for time in self.data.keys():
            if item not in self.data[time]:
                self.data[time][item] = None

    def register_items(self, items, prefix=None):
        for item in items:
            self.register_item(item)

    def add_data(self, time, item, data):
        if time not in self.data.keys():
            self.data[time] = dict()
        self.data[time][item] = data

    def add_data_items(self, time, items, datas):
        for item, data in zip(items, datas):
            self.add_data(time, item, data)

    def get_data(self, item, time_frame=None):
        times = sorted(self.data.keys())
        time_frame = [times[0], times[-1]] if time_frame is None else time_frame
        ret_val = list()
        for time in times:
            if time_frame[0] <= time <= time_frame[1]:
                ret_val.append(self.data[time][item])
        return np.array(ret_val)

    def get_all_data(self):
        return self.data

    def export_csv(self, fields, filepath):
        pass

    def simple_plot(self, items, time_frame=None, subplot=False):
        if isinstance(items, str):
            items = [items]
        plt.figure()
        if subplot:
            plt.subplot(len(items), 1, 1)
        times = np.array(sorted(self.data.keys()))
        if time_frame is not None:
            tmp_times = list()
            for time in times:
                if time_frame[0] <= time <= time_frame[1]:
                    tmp_times.append(time)
            times = np.array(tmp_times)

        for ind, item in enumerate(items, start=1):
            data = self.get_data(item, time_frame)
            if subplot:
                plt.subplot(len(items), 1, ind)
                plt.grid(True)
                plt.title(item)
            plt.plot(times, data, label=item)
        if not subplot:
            plt.legend()
        plt.grid()
import numpy as np

class AeroTable(object):
    def __init__(self, aoa, cl, cd):
        self.aoa = aoa

        self.aoa[self.aoa > 180] = self.aoa[self.aoa > 180] - 360
        self.aoa[self.aoa <= -180] = self.aoa[self.aoa <= -180] + 360

        self.cl = cl
        self.cd = cd

    def __getitem__(self, item):
        idx = np.abs(self.aoa - item).argmin()
        indm = idx-1
        indp = (idx+1) % (len(self.aoa)-1)
        aoa_cen = self.aoa[idx]
        aoa_m1 = self.aoa[indm]
        aoa_p1 = self.aoa[indp]

        cl = np.interp(item, [aoa_m1, aoa_cen, aoa_p1], [self.cl[indm], self.cl[idx], self.cl[indp]])
        cd = np.interp(item, [aoa_m1, aoa_cen, aoa_p1], [self.cd[indm], self.cd[idx], self.cd[indp]])

        return cl, cd
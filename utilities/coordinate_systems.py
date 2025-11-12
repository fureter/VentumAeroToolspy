import numpy as np

def rotation_matrix(angles):
        roll = angles[6]
        pitch = angles[7]
        yaw = angles[8]
        cp = np.cos(pitch)
        cr = np.cos(roll)
        cy = np.cos(yaw)
        sp = np.sin(pitch)
        sr = np.sin(roll)
        sy = np.sin(yaw)
        return np.matrix([[cy * cp, sy * cp, -sp],
                          [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr],
                          [cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr]])

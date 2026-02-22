import numpy as np
import pyproj
from scipy.spatial.transform import Rotation

ECEF_PROJ = pyproj.CRS(proj='geocent', ellps='WGS84', datum='WGS84')
LLA_PROJ = pyproj.CRS(proj='latlong', ellps='WGS84', datum='WGS84')


def rotation_matrix(angles):
        roll = angles[0]
        pitch = angles[1]
        yaw = angles[2]

        cy = np.cos(yaw)
        sy = np.sin(yaw)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cr = np.cos(roll)
        sr = np.sin(roll)

        # return Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
        return np.array([[cp*cy, cp*sy, -sp],
                         [sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, sr*cp],
                         [cr*sp*cy + sr*sy, cr*sp*sy - sr*cy, cr*cp]]).T

def euler_to_quaternion(angles):
    roll = angles[0]
    pitch = angles[1]
    yaw = angles[2]

    q0 = np.cos(roll/2)*np.cos(pitch/2)*np.cos(yaw/2) + np.sin(roll/2)*np.sin(pitch/2)*np.sin(yaw/2)
    q1 = np.sin(roll/2)*np.cos(pitch/2)*np.cos(yaw/2) - np.cos(roll/2)*np.sin(pitch/2)*np.sin(yaw/2)
    q2 = np.cos(roll/2)*np.sin(pitch/2)*np.cos(yaw/2) + np.sin(roll/2)*np.cos(pitch/2)*np.sin(yaw/2)
    q3 = np.cos(roll/2)*np.cos(pitch/2)*np.sin(yaw/2) - np.sin(roll/2)*np.sin(pitch/2)*np.cos(yaw/2)

    return np.array([q0, q1, q2, q3])


def enu_to_ecef(lla, val):
    lat = lla[0]
    lon = lla[1]
    clam = np.cos(lon)
    slam = np.sin(lon)
    cphi = np.cos(lat)
    sphi = np.sin(lat)
    dcm = np.array([[-slam, -clam*sphi, clam*cphi],
                     [clam, -slam*sphi, slam*cphi],
                     [0, cphi, sphi]])
    return dcm @ val

def ecef_to_enu(lla, val):
    lat = lla[0]
    lon = lla[1]
    clam = np.cos(lon)
    slam = np.sin(lon)
    cphi = np.cos(lat)
    sphi = np.sin(lat)
    dcm = np.array([[-slam, -clam*sphi, clam*cphi],
                     [clam, -slam*sphi, slam*cphi],
                     [0, cphi, sphi]]).T
    return dcm @ val

def ned_to_ecef(lla, val):
    lat = lla[0]
    lon = lla[1]
    clam = np.cos(lon)
    slam = np.sin(lon)
    cphi = np.cos(lat)
    sphi = np.sin(lat)
    dcm = np.array([[-sphi*clam, -sphi*slam, cphi],
                     [-slam, clam, 0],
                     [-cphi*clam, -cphi*slam, -sphi]])
    return dcm.T @ val

def ecef_to_ned(lla, val):
    lat = lla[0]
    lon = lla[1]
    clam = np.cos(lon)
    slam = np.sin(lon)
    cphi = np.cos(lat)
    sphi = np.sin(lat)
    dcm = np.array([[-sphi*clam, -sphi*slam, cphi],
                     [-slam, clam, 0],
                     [-cphi*clam, -cphi*slam, -sphi]])
    return dcm.T @ val

def _pyproj_transform(projection1, projection2):
    return pyproj.Transformer.from_crs(projection1, projection2)

def lla_to_ecef(lla):
    x, y, z = _pyproj_transform(LLA_PROJ, ECEF_PROJ).transform(lla[1], lla[0], lla[2], radians=True)

    return x, y, z

def ecef_to_lla(ecef):
    lat, lon, alt = _pyproj_transform(ECEF_PROJ, LLA_PROJ).transform(ecef[0], ecef[1], ecef[2], radians=True)

    return np.array([lat, lon, alt])

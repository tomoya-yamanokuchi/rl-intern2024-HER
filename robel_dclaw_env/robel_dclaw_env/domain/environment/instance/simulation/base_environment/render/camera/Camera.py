import numpy as np
from mujoco_py.modder import CameraModder
from transforms3d.euler import euler2quat
from .CameraArgDict import CameraArgDict


class Camera:
    def __init__(self, cameraArgDict: CameraArgDict):
        self.camera_modder    = CameraModder(cameraArgDict['sim'])
        self.x_coordinate     = cameraArgDict['x_coordinate']
        self.y_coordinate     = cameraArgDict['y_coordinate']
        self.z_distance       = cameraArgDict['z_distance']
        self.orientation      = cameraArgDict['orientation']

    def set_camera_posture(self, camera_name: str):
        self.__set_xyz_position(camera_name)
        self.__set_quaternion(camera_name)
        self.camera_modder.sim.step()

    def __set_xyz_position(self, camera_name):
        self.camera_modder.set_pos(
            name  = camera_name,
            value = [
                self.x_coordinate,
                self.y_coordinate,
                self.z_distance,
            ],
        )

    def __set_quaternion(self, camera_name):
        self.camera_modder.set_quat(
            name  = camera_name,
            value = euler2quat(np.deg2rad(self.orientation), 0.0, np.pi/2)
        )

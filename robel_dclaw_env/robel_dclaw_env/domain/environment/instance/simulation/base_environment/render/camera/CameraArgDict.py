from typing import TypedDict
from mujoco_py import MjSim


class CameraArgDict(TypedDict):
    sim              : MjSim
    x_coordinate     : float
    y_coordinate     : float
    z_distance       : float
    orientation      : float

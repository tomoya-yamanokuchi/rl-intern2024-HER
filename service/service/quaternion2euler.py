import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion2euler(quaternion: np.ndarray):
    assert quaternion.shape == (4,)
    rotation     = R.from_quat(quaternion)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    return euler_angles

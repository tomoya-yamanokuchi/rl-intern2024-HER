import numpy as np
from pprint import pprint


class RobotCanonicalRGB:
    def __init__(self):
        self.floor_rgb  = np.array([255, 255, 255], dtype=np.uint8)
        self.robot_rgb  = np.array([ 38,  38,  38], dtype=np.uint8)
        self.finger_rgb = np.array([255,  85,   0], dtype=np.uint8)


    def get_rgb_dict(self):
        return {
            **self.__floor(),
            **self.__claw1(),
            **self.__claw2(),
            **self.__claw3(),
        }

    def __floor(self):
        return {"floor" : self.floor_rgb}

    def __claw1(self):
        return {
            # --> claw1
            "FFbase_xh28"               : self.robot_rgb,
            "FF10_metal_clamping"       : self.robot_rgb,
            "FF10_metal_clamping_small" : self.robot_rgb,
            "FF10_xh28"                 : self.robot_rgb,
            "FFL11_metal_clamping_small": self.robot_rgb,
            "FFL11_xh28"                : self.robot_rgb,
            "FFL11_metal_clamping"      : self.robot_rgb,
            "FFL12_metal_clamping"      : self.robot_rgb,
            "FFL12_plastic_finger"      : self.finger_rgb,
        }

    def __claw2(self):
        return {
            # --> claw2
            "MFbase_xh28"               : self.robot_rgb,
            "MF20_metal_clamping"       : self.robot_rgb,
            "MF20_metal_clamping_small" : self.robot_rgb,
            "MF20_xh28"                 : self.robot_rgb,
            "MFL21_metal_clamping_small": self.robot_rgb,
            "MFL21_xh28"                : self.robot_rgb,
            "MFL21_metal_clamping"      : self.robot_rgb,
            "MFL22_metal_clamping"      : self.robot_rgb,
            "MFL22_plastic_finger"      : self.finger_rgb,
        }

    def __claw3(self):
        return {
            # --> claw3
            "THbase_xh28"               : self.robot_rgb,
            "TH30_metal_clamping"       : self.robot_rgb,
            "TH30_metal_clamping_small" : self.robot_rgb,
            "TH30_xh28"                 : self.robot_rgb,
            "THL31_metal_clamping_small": self.robot_rgb,
            "THL31_xh28"                : self.robot_rgb,
            "THL31_metal_clamping"      : self.robot_rgb,
            "THL32_metal_clamping"      : self.robot_rgb,
            "THL32_plastic_finger"      : self.finger_rgb,
        }


if __name__ == '__main__':
    can = RobotCanonicalRGB()
    pprint(can.get_rgb_dict())

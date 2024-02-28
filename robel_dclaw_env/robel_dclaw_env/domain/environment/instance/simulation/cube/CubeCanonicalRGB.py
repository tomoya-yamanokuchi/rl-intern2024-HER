from pprint import pprint
import numpy as np
from robel_dclaw_env.domain.environment.instance.simulation.base_environment import RobotCanonicalRGB


class CubeCanonicalRGB:
    def __init__(self):
       self.robot_canonical_rgb = RobotCanonicalRGB()

    def get_rgb_dict(self, object_rgb, num_object_geom):
        return {
            **self.robot_canonical_rgb.get_rgb_dict(),
            **self.__object()
        }

    def __object(self):
        # import ipdb; ipdb.set_trace()
        # assert len(object_rgb) == num_object_geom
        rgb_dict = {
            "object_geom_vis_dice_green"    : [68, 201, 117],
            "object_geom_vis_dice_red"      : [255, 50, 50],
            "object_geom_vis_dice_blue"     : [50, 132, 255],
            "object_geom_vis_dice_yellow"   : [255, 255, 0],
            "object_geom_vis_dice_orange"   : [255, 178, 56],
            "object_geom_vis_dice_purple"   : [141, 75, 221],
        }
        # import ipdb; ipdb.set_trace()
        return rgb_dict

if __name__ == '__main__':

    object_rgb      = [255, 0, 0]
    num_object_geom = 30

    can = CubeCanonicalRGB()
    pprint(can.get_rgb_dict(object_rgb, num_object_geom))

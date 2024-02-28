from pprint import pprint
import numpy as np
from robel_dclaw_env.domain.environment.instance.simulation.base_environment import RobotCanonicalRGB


class CubeCanonicalRGB:
    def __init__(self):
       self.robot_canonical_rgb = RobotCanonicalRGB()

    def get_rgb_dict(self, object_rgb, num_object_geom):
        return {
            **self.robot_canonical_rgb.get_rgb_dict(),
            **self.__object(object_rgb, num_object_geom)
        }

    def __object(self, object_rgb: list, num_object_geom):
        # import ipdb; ipdb.set_trace()
        assert len(object_rgb) == num_object_geom
        rgb_dict = {}
        for i in range(num_object_geom):
            rgb_dict["object_geom_vis_{}".format(i)] = object_rgb[i]
        # print(rgb_dict)
        # import ipdb; ipdb.set_trace()
        return rgb_dict

if __name__ == '__main__':

    object_rgb      = [255, 0, 0]
    num_object_geom = 30

    can = CubeCanonicalRGB()
    pprint(can.get_rgb_dict(object_rgb, num_object_geom))

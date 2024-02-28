import copy
import numpy as np
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.BaseEnvironment import BaseEnvironment
from robel_dclaw_env.domain.environment.kinematics import NumpyForwardKinematics
from robel_dclaw_env.domain.environment.instance.simulation.base_environment import EndEffectorPosition
from task_space import AbstractTaskSpaceTransformer
from torch_numpy_converter import NTD, to_tensor, to_numpy, to_tensor_without_cuda
from .AbstractStateFactory import AbstractStateFactory
from .EndEffectorPosition import EndEffectorPosition
from .AbstractObjectStateDimensionOfInterest import AbstractObjectStateDimensionOfInterest


class GetSurfaceCoordinates:
    def __init__(self,
            sim        : BaseEnvironment,
            # doi_object_state: AbstractObjectStateDimensionOfInterest,
        ):
        self.sim           = sim
        # self.doi_object_state   = doi_object_state

    def get_surface_coords(self):
        state       = self.sim.get_state()
        # ---
        box_geom_id = self.sim.model.geom_name2id('object_geom_vis_0')
        box_pos     = self.sim.data.geom_xpos[box_geom_id]
        box_size    = self.sim.model.geom_size[box_geom_id]

        # import ipdb; ipdb.set_trace()
        # ボックスの各面の座標を計算
        # ここでは簡単のため、ボックスが軸に沿っていると仮定
        box_surface_coords = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    coord = box_pos + np.multiply(box_size, [dx/2, dy/2, dz/2])
                    box_surface_coords.append(coord)

        # 座標の出力
        for coord in box_surface_coords:
            print(coord)

        # -------
        for i, coord in enumerate(box_surface_coords):
            site_id = self.sim.model.site_name2id(f'site{i+1}')
            self.sim.model.site_pos[site_id] = coord


        # import ipdb; ipdb.set_trace()
        return state

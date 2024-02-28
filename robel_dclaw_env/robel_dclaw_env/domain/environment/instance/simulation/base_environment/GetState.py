import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.BaseEnvironment import BaseEnvironment
from robel_dclaw_env.domain.environment.kinematics import NumpyForwardKinematics
from robel_dclaw_env.domain.environment.instance.simulation.base_environment import EndEffectorPosition
from task_space import AbstractTaskSpaceTransformer
from torch_numpy_converter import NTD, to_tensor, to_numpy, to_tensor_without_cuda
from service import quaternion2euler
from .AbstractStateFactory import AbstractStateFactory
from .EndEffectorPosition import EndEffectorPosition
from .AbstractObjectStateDimensionOfInterest import AbstractObjectStateDimensionOfInterest


class GetState:
    def __init__(self,
            base_env        : BaseEnvironment,
            StateFactory    : AbstractStateFactory,
            task_space      : AbstractTaskSpaceTransformer,
            # doi_object_state: AbstractObjectStateDimensionOfInterest,
        ):
        self.base_env           = base_env
        self.StateFactory       = StateFactory
        self.task_space         = task_space
        self.forward_kinematics = NumpyForwardKinematics()
        # self.doi_object_state   = doi_object_state

    def get_state(self):
        state                 = self.base_env.get_state()
        robot_position        = state.qpos[:9]
        end_effector_position = self.forward_kinematics.calc(robot_position).squeeze()
        task_space_position   = self.task_space.end2task(EndEffectorPosition(NTD(end_effector_position)))
        # ------------------
        body_id      = self.base_env.model.body_name2id('pushing_object') # ボディのIDを取得
        quaternion   = self.base_env.data.body_xquat[body_id]             # ボディのクォータニオンを取得
        euler_angles = quaternion2euler(quaternion) # クォータニオンをオイラー角に変換
        # print("euler_angles = {}".format(euler_angles))
        # print("euler_angles (z) = {}".format(euler_angles[0]))
        # ----------------
        state = self.StateFactory.create_for_get_state(
            robot_position        = robot_position,
            # object_position       = self.doi_object_state.extract(state.qpos),
            object_position       = state.qpos[18:],
            object_rotation       = euler_angles,
            robot_velocity        = state.qvel[:9],
            # object_velocity       = self.doi_object_state.extract(state.qvel),
            object_velocity       = state.qvel[18:],
            end_effector_position = end_effector_position,
            task_space_position   = task_space_position,
            time                  = state.time,
            act                   = state.act,
            udd_state             = state.udd_state,
        )
        # import ipdb; ipdb.set_trace()
        return state

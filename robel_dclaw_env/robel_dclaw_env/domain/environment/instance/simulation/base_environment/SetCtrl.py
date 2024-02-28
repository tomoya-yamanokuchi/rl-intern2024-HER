import numpy as np
import copy
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.BaseEnvironment import BaseEnvironment
from robel_dclaw_env.domain.environment.kinematics import NumpyInverseKinematics
from task_space import AbstractTaskSpaceTransformer
from .ReturnCtrl import ReturnCtrl
from .JointPosition import JointPosition
from torch_numpy_converter import to_numpy
from .CtrlDict import CtrlDict


class SetCtrl:
    def __init__(self, base_env: BaseEnvironment, task_space: AbstractTaskSpaceTransformer):
        self.base_env           = base_env
        self.task_space         = task_space
        self.inverse_kinematics = NumpyInverseKinematics()

    def set_ctrl(self, task_space_position: object) -> CtrlDict:
        ctrl_end_effector = self.task_space.task2end(task_space_position)
        ctrl_joint        = self.inverse_kinematics.calc(ctrl_end_effector.value.squeeze(axis=0))
        self.base_env.set_ctrl(ctrl_joint.squeeze())
        # ---------------
        # print("task_space_position = ", np.max(task_space_position.value.squeeze()))
        # ----------------
        return CtrlDict(
            task_space_abs_position  = task_space_position,
            # task_space_diff_position = None, # モデル学習には使えないため間違って使わないように
            end_effector_position    = ctrl_end_effector,
            joint_space_position     = JointPosition(ctrl_joint.squeeze()),
        )

    def set_joint_space_ctrl(self, joint_space_position: object):
        self.base_env.set_ctrl(joint_space_position.squeeze())
        return self.base_env.ctrl

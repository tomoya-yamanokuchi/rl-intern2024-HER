from typing import TypedDict
from .EndEffectorPosition import EndEffectorPosition
from .JointPosition import JointPosition


class CtrlDict(TypedDict):
    task_space_abs_position  : object
    task_space_diff_position : object
    end_effector_position    : EndEffectorPosition
    joint_space_position     : JointPosition

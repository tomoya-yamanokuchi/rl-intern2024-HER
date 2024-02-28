import numpy as np

from .object_state import PushingPosition, PushingRotation, PushingVelocity
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.state.MjSimTime import MjSimTime
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.state.MjSimAct import MjSimAct
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.state.MjSimUddState import MjSimUddState
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.state.RobotPosition import RobotPosition
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.state.RobotVelocity import RobotVelocity
from robel_dclaw_env.domain.environment.kinematics import EndEffectorPosition

# from ..base_environment.StateDict import
from ..base_environment import StateDict, State
from torch_numpy_converter import to_tensor_without_cuda, NTD


class CubeStateFactory:
    @staticmethod
    def create_for_init_env(TaskSpacePosition, task_space_position, robot_velocity, object_position, object_rotation, object_velocity, time, act, udd_state):
        return StateDict(
            task_space_position   = TaskSpacePosition(NTD(np.array(task_space_position))),
            end_effector_position = None,
            robot_position        = None,
            robot_velocity        = RobotVelocity(np.array(robot_velocity)),
            object_position       = PushingPosition(np.array(object_position)),
            object_rotation       = PushingRotation(np.array(object_rotation)),
            object_velocity       = PushingVelocity(np.array(object_velocity)),
            time                  = MjSimTime(time),
            act                   = MjSimAct(act),
            udd_state             = MjSimUddState(udd_state),
        )

    @staticmethod
    def create_for_get_state(task_space_position, end_effector_position, robot_position, robot_velocity, object_position, object_rotation, object_velocity, time, act, udd_state):
        return StateDict(
            task_space_position   = task_space_position,
            end_effector_position = EndEffectorPosition(end_effector_position),
            robot_position        = RobotPosition(np.array(robot_position)),
            robot_velocity        = RobotVelocity(np.array(robot_velocity)),
            object_position       = PushingPosition(np.array(object_position)),
            object_rotation       = PushingRotation(np.array(object_rotation)),
            object_velocity       = PushingVelocity(np.array(object_velocity)),
            time                  = MjSimTime(time),
            act                   = MjSimAct(act),
            udd_state             = MjSimUddState(udd_state),
        )

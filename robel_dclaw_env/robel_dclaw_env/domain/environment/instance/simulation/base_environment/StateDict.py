from typing import TypedDict


class StateDict(TypedDict):
    task_space_position   : object
    end_effector_position : object
    robot_position        : object
    robot_velocity        : object
    object_position       : object
    object_rotation       : object
    object_velocity       : object
    time                  : object
    act                   : object
    udd_state             : object

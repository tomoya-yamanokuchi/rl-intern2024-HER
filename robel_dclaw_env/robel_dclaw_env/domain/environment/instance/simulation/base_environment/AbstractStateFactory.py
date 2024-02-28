from abc import ABC, abstractmethod

class AbstractStateFactory(ABC):
    @abstractmethod
    def create_for_init_env(
            task_space_position,
            robot_velocity,
            object_position,
            object_rotation,
            object_velocity,
            time,
            act,
            udd_state
            ) -> object:
        pass

    @abstractmethod
    def create_for_get_state(
            task_space_position,
            end_effector_position,
            robot_position,
            robot_velocity,
            object_position,
            object_rotation,
            object_velocity,
            time,
            act,
            udd_state
            ) -> object:
        pass

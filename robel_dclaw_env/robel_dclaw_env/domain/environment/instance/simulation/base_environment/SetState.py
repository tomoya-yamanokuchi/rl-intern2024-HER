import numpy as np
from mujoco_py import MjSimState
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.domain.environment.kinematics import NumpyInverseKinematics
# from task_space import AbstractTaskSpaceTransformer
from torch_numpy_converter import NTD, to_tensor, to_numpy
from .StateDict import StateDict


class SetState:
    def __init__(self, sim, task_space):
        self.sim                  = sim
        self.task_space           = task_space
        self.inverse_kinematics   = NumpyInverseKinematics()


    def set_state(self, state: StateDict):
        new_state = MjSimState(
            time      = state["time"].value,
            qpos      = self._set_qpos(state),
            qvel      = self._set_qvel(state),
            act       = state["act"].value,
            udd_state = dict(state["udd_state"].value),
        )
        self.sim.set_state(new_state)
        self.sim.forward()


    def _set_qpos(self, state: StateDict):
        # -------------
        # state["task_space_position"].value = to_tensor(state["task_space_position"].value)
        # import ipdb; ipdb.set_trace()
        end_effector_position = self.task_space.task2end(state["task_space_position"])
        joint_position        = self.inverse_kinematics.calc(end_effector_position.value.squeeze(0))
        # ----------
        qpos      = np.zeros(self.sim.model.nq)
        qpos[:9]  = joint_position.squeeze()
        qpos[18:] = state["object_position"].value.squeeze() # <--- env specific!
        return qpos


    def _set_qvel(self, state: StateDict):
        qvel      = np.zeros(self.sim.model.nv)
        qvel[:9]  = state["robot_velocity"].value.squeeze()
        qvel[18:] = state["object_velocity"].value.squeeze()
        return qvel

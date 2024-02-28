import os
import mujoco_py
# ----- general module -----
from ....kinematics import NumpyForwardKinematics
from ....kinematics import NumpyInverseKinematics
from ..base_environment import AbstractEnvironment, BaseEnvironment, ModelLoader
from ..base_environment import SetState, GetState, StateDict, GetSurfaceCoordinates
from ..base_environment import SetCtrl, CtrlDict
from ..base_environment.dynamics_parameter import RobotDynamicsParameter
from ..base_environment.joint_range import RobotJointRange
from ..base_environment.ctrl_range import RobotCtrlRange
from task_space import TaskSpaceFactory
from service import NTD
# ----- Env Specific ---
from .PushingObjectDyanmicsParameter import PushingObjectDyanmicsParameter
from .PushingObjectJointRange import PushingObjectJointRange
from .PushingTarget import PushingTarget
from .CubeStateFactory import CubeStateFactory as StateFactory
from task_space.end_effector_2d import BiasedEndEffectorPosition_2D_Plane
from .CubeCanonicalRGB import CubeCanonicalRGB
from .get_object_geom_names import get_object_geom_names


class CubeSimulationEnvironment(AbstractEnvironment):
    def __init__(self, config, use_render=True):
        self.env_name               = "sim_cube"
        self.canonical_rgb_manager  = CubeCanonicalRGB()
        # ----
        self.base_env               = BaseEnvironment(config, use_render)
        self.forward_kinematics     = NumpyForwardKinematics()
        self.inverse_kinematics     = NumpyInverseKinematics()
        self.config                 = config
        self.task_space_transformer = TaskSpaceFactory.create_transformer(self.env_name, mode="numpy")
        self.TaskSpaceValueObject   = TaskSpaceFactory.create_position(self.env_name)

    def set_xml_path(self, xml_path: str):
        self.xml_path = xml_path

    def load_model(self):
        # assert self.xml_str is not None
        print("self.xml_path = {}".format(self.xml_path))
        # import ipdb; ipdb.set_trace()
        self.model             = self.base_env.load_model_from_path(self.xml_path)
        sim                    = mujoco_py.MjSim(self.model)
        self.base_env.set_sim(sim)
        self.base_env.build_viewer()
        self.base_env.view()
        self.object_geom_names = get_object_geom_names(self.model, self.config.texture.task_relevant_geom_group_name)
        self.canonical_rgb     = self.canonical_rgb_manager.get_rgb_dict(self.config.xml.rgb.object, num_object_geom=len(self.object_geom_names))

        self.base_env.build_render(self.canonical_rgb)
        self.setState          = SetState(self.base_env.sim, self.task_space_transformer)
        self.getState          = GetState(self.base_env.sim, StateFactory, self.task_space_transformer)
        # self.surfaceCoords     = GetSurfaceCoordinates(self.base_env.sim)
        self.setCtrl           = SetCtrl(self.base_env, self.task_space_transformer)
        # self.setTargetPosition = PushingTarget(self.base_env.sim)
        # self.setTargetPosition.set_target_visible(self.config.target.visible)
        RobotDynamicsParameter(self.base_env.sim).set(self.config.dynamics.robot)
        PushingObjectDyanmicsParameter(self.base_env.sim).set(self.config.dynamics.object)
        RobotJointRange(self.base_env.sim).set_range(**self.config.joint_range.robot)
        PushingObjectJointRange(self.base_env.sim).set_range(**self.config.joint_range.object)
        RobotCtrlRange(self.base_env.sim).set_range()

    def reset(self, state: StateDict, xml_path: str = None, verbose: bool=False):
        if xml_path is None:
            xml_path = os.path.join(self.config.model_dir, self.config.model_file)
        if self.base_env.sim is None:
            import ipdb; ipdb.set_trace()
            self.load_model()
        self.base_env.sim.reset()
        self.set_state(state)
        self.base_env.rendering_manager.register_new_randomized_texture_collection()
        # self.base_env.step()
        # for i in range(100):
            # self.base_env.sim.step()

        # import ipdb; ipdb.set_trace()
        self.set_task_space_ctrl(self.TaskSpaceValueObject(NTD(state["task_space_position"].value)))
        self.step()
        # self.base_env.sim.step()


        if verbose: print("\n <---- reset --->\n")

    def render(self):
        return self.base_env.render()

    def view(self):
        self.base_env.view()

    def get_state(self) -> StateDict:
        return self.getState.get_state()

    # def get_surface_coords(self):
    #     self.surfaceCoords.get_surface_coords()

    def set_state(self, state: StateDict):
        self.setState.set_state(state)

    def set_task_space_ctrl(self, TaskSpacePosition: BiasedEndEffectorPosition_2D_Plane) -> CtrlDict:
        return self.setCtrl.set_ctrl(TaskSpacePosition)

    def set_joint_space_ctrl(self, joint_space_position):
        return self.setCtrl.set_joint_space_ctrl(joint_space_position)

    def set_target_position(self, target_position):
        self.setTargetPosition.set_target_position(target_position)

    def step(self, is_view=False):
        # import ipdb; ipdb.set_trace()
        self.base_env.step(is_view)

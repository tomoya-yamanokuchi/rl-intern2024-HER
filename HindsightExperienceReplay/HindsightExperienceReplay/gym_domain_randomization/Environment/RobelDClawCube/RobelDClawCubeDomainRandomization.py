import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

from .RobelDClawCubeDomainInfo import RobelDClawCubeDomainInfo
from HindsightExperienceReplay import UserDefinedSettingsFactory

from robel_dclaw_env.domain.environment.instance.simulation.cube.CubeSimulationEnvironment import CubeSimulationEnvironment as Env
from domain_object_builder import DomainObject
from service import NTD


class RobelDClawCubeDomainRandomization(gym.Env):
    """
    物理パラメータをランダマイズ
    物理パラメータをエピソードごとにランダマイズして学習　OR 各ドメインごとに学習
    （ドメインの変化の影響を確認するために，1つのドメインで学習した方策に対して，どれくらいドメインを変更すると学習に影響があるのかを確認）
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, domain_object: DomainObject, userDefinedSettings: UserDefinedSettingsFactory, domain_range=None):

        self.env : Env                = domain_object.env
        self.init_state               = domain_object.init_state
        self.TaskSpaceValueObject     = domain_object.TaskSpaceValueObject
        self.TaskSpaceDiffValueObject = domain_object.TaskSpaceDiffValueObject
        self.xml_path                 = domain_object.original_xml_path
        self.task_space_position_init = self.init_state["task_space_position"]

        # import ipdb; ipdb.set_trace()
        # -------------------------------------------------

        self.userDefinedSettings = userDefinedSettings
        self.domainInfo = RobelDClawCubeDomainInfo(userDefinedSettings, domain_range)

        self.max_speed = 8  # 8
        self.task_space_max_val = 1.0
        # self.dt = .05
        # self.g = 10.
        # self.m = 1.
        # self.l = 1.

        self.viewer = None


        self.action_space = spaces.Box(low=-self.task_space_max_val, high=self.task_space_max_val, shape=(6,))


        high = np.array([1., 1., self.max_speed])
        self.observation_space = spaces.Box(low=-high, high=high)

        '''
        # 原点位置でz軸方向に45度の向きにもっていく
        # ideal_goal = [x, y, x, z_degree, x_degree?, y_degree?]
        '''
        self.ideal_goal = np.array([0, 0, 0, 45, 0., 0.])

        self._seed()

        self.achievement_threshold = 0.1  # 0.1
        # import ipdb; ipdb.set_trace()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, u):
        # import ipdb; ipdb.set_trace()
        domain_parameter = self.domainInfo.get_domain_parameters()
        # ---
        reward           = self.calc_reward()
        task_achievement = self.check_task_achievement()
        # ----
        self.env.set_task_space_ctrl(self.TaskSpaceValueObject(NTD(u))) # 差分制御にしないと多分ダメ
        self.env.step() # inplicit_stepの幅の調整も多分必用
        self.state = self.env.get_state()
        # ---　固定 ---
        observation = self._get_obs()
        done = False
        self.step_num += 1  # 終わってから追加
        return observation, reward, done, domain_parameter, task_achievement

    def calc_reward(self):
        '''
        ・一旦借り置きで常に0を返す
        ・タスク評価に報酬を使って解析する場合には要追加
        '''
        pseudo_costs = 0.0
        costs        = pseudo_costs
        reward       = -costs
        return reward


    def check_task_achievement(self):
        object_xyz_coordinates = self.state['object_position'].value.squeeze()[:3]
        object_rotation        = self.state['object_rotation'].value.squeeze()[:3]
        # ---
        eval_state = np.concatenate([object_xyz_coordinates, object_rotation], axis=0)
        goal_norm  = np.linalg.norm(eval_state - self.ideal_goal)
        # --- 固定 ---
        if goal_norm < self.achievement_threshold:
            task_achievement = 1.
        else:
            task_achievement = 0.
        # ---
        return task_achievement


    def get_domain_parameter_dim(self):
        return self.domainInfo.get_domain_parameters().shape[0]


    def reset(self):
        self.step_num = 0
        self.last_u = None
        # ---
        self.env.set_xml_path(self.xml_path)
        self.env.load_model()
        self.env.reset(self.init_state)
        self.env.set_task_space_ctrl(self.task_space_position_init)
        self.env.step(is_view=True)
        self.env.render()
        # ---
        self.state = self.env.get_state()
        # ---
        # import ipdb; ipdb.set_trace()
        return self._get_obs()


    def _get_obs(self):
        task_space_robot_position = self.state['task_space_position'].value.squeeze()
        object_xyz_coordinates    = self.state['object_position'].value.squeeze()[:3]
        object_rotation           = self.state['object_rotation'].value.squeeze()[:3]
        # ----
        obs = {}
        # ---
        obs['observation'] = np.concatenate([
            task_space_robot_position, # ロボット手先座標位置(2次元 x 3本 = 6次元)
            object_xyz_coordinates,    # サイコロのxyz座標 (3次元)
            object_rotation,           # サイコロの回転角度 (3次元)
        ], axis=0)
        # ---
        # import ipdb; ipdb.set_trace()
        obs['achieved_goal'] = np.concatenate([
            task_space_robot_position, # ロボット手先座標位置(2次元 x 3本 = 6次元)
            object_xyz_coordinates,    # サイコロのxyz座標 (3次元)
            object_rotation,           # サイコロの回転角度 (3次元)
        ], axis=0)
        # ---
        obs['desired_goal']  = self.ideal_goal
        return obs


    def _render(self, mode='human', close=False):

        self.env.view()
        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

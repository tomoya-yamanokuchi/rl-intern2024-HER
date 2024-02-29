import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

from .RobelDClawCubeDomainInfo import RobelDClawCubeDomainInfo
from HindsightExperienceReplay import UserDefinedSettingsFactory

from robel_dclaw_env.domain.environment.instance.simulation.cube.CubeSimulationEnvironment import CubeSimulationEnvironment as Env
from domain_object_builder import DomainObject


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

        self.env                      = domain_object.env
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
        self.max_torque = 2.  # 2.
        # self.dt = .05
        # self.g = 10.
        # self.m = 1.
        # self.l = 1.

        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self.ideal_goal = np.array([1., 0., 0.])  # checked [y,x,-]　頂上が0度の設定

        self._seed()

        self.achievement_threshold = 0.1  # 0.1
        # import ipdb; ipdb.set_trace()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, u):
        th, thdot = self.state  # th := theta

        domain_parameter = self.domainInfo.get_domain_parameters()
        dt, g, m, l, torque_weight, torque_bias = domain_parameter

        u = u * torque_weight + torque_bias  # for using domain randomization

        if self.domainInfo.torque_limit:
            u = np.clip(u, -self.max_torque, self.max_torque)
        else:
            # u = u[0]
            u = u

        self.last_u = u  # for rendering
        reward = self.calc_reward(th, thdot, u)
        task_achievement = self.check_task_achievement(th, thdot, u)

        newthdot = thdot + (-3 * g / (2 * l) *
                            np.sin(th + np.pi) + 3. / (m * l**2) * u) * dt
        newth = th + newthdot * dt
        if self.domainInfo.velocity_limit:
            newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])

        observation = self._get_obs()
        done = False
        self.step_num += 1  # 終わってから追加
        return observation, reward, done, domain_parameter, task_achievement



    def calc_reward(self, th, thdot, u):
        ver = 1  # 0:origin, over1:mine
        if ver == 0:
            costs = angle_normalize(th)**2 + 0.1 * (thdot**2) + 0.001 * (u**2)
        elif ver == 1:
            costs = angle_normalize(th)**2 + 0.1 * abs(thdot) + 0.001 * abs(u)

        reward = -costs
        return reward

    def check_task_achievement(self, th, thdot, u):
        theta, thetadot = th, thdot

        state = np.array([np.cos(theta), np.sin(theta), thetadot])

        goal_norm = np.linalg.norm(state-self.ideal_goal)

        if goal_norm < self.achievement_threshold:
            task_achievement = 1.
        else:
            task_achievement = 0.

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
        task_space_robot_position = self.state['task_space_position']
        dice_position             = self.state['object_position']
        dice_rotation             = self.state['object_rotation']

        import ipdb; ipdb.set_trace()
        # ----
        obs = {}
        # ---
        obs['observation']   = np.array([])
        # ---
        obs['achieved_goal'] = np.array([np.cos(theta), np.sin(theta), thetadot])
        obs['desired_goal']  = self.ideal_goal
        return obs

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            fname = path.join(path.dirname(__file__), "clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

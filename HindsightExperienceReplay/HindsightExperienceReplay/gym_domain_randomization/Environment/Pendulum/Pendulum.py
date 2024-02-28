# import gym
import time
import numpy as np

from .PendulumDomainRandomization import PendulumDomainRandomization


class Pendulum(object):
    """
    default max step length: 200 -> learning max step length: 150
    """

    def __init__(self, userDefinedSettings, domain_range=None):
        self.userDefinedSettings = userDefinedSettings
        self.env = PendulumDomainRandomization(
            userDefinedSettings, domain_range)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.DOMAIN_PARAMETER_DIM = self.env.get_domain_parameter_dim()
        self.RENDER_INTERVAL = 0.05  # [s]
        self.MAX_EPISODE_LENGTH = 150
        self.ACTION_MAPPING_FLAG = False  # True

        self.ACTION_DISCRETE_FLAG = userDefinedSettings.ACTION_DISCRETE_FLAG
        self.STATE_DIM = self.env.observation_space.shape[0]
        if self.ACTION_DISCRETE_FLAG:
            self.actions = [-1, -0.5, -0.2, 0, 0.2, 0.5, 1]
            self.ACTION_DIM = len(self.actions)
        else:
            self.ACTION_DIM = self.env.action_space.shape[0]

        self.MAX_ACTION = self.env.max_torque

        self.domainInfo = self.env.domainInfo

    def reset(self, get_domain_parameter=False):
        state = self.env.reset()
        if get_domain_parameter:
            domain_parameter = self.domainInfo.get_domain_parameters()
            return state, domain_parameter
        else:
            return state

    def step(self, action, get_task_achievement=False):
        if self.ACTION_DISCRETE_FLAG:
            action = self.mapping_action_discrete2continues(action)
        if self.ACTION_MAPPING_FLAG:
            action = self.mapping_action(action)
        next_state, reward, done, domain_parameter, task_achievement = self.env.step(
            action)

        info = {'domain_parameter': domain_parameter,
                'is_success': task_achievement}
        return next_state, reward, done, info

    def compute_reward(self, transitions):
        goal_norm = np.linalg.norm((transitions['ag_next']-transitions['g']), axis=1)
        reward = np.where(goal_norm < self.env.achievement_threshold, 1., 0.)
        return reward

    def random_action_sample(self):
        action = self.env.action_space.sample()
        if self.ACTION_MAPPING_FLAG:
            low = self.env.action_space.low
            high = self.env.action_space.high
            action = 2 * (action - low) / (high - low) - 1
        return action

    def render(self):
        # import ipdb; ipdb.set_trace()
        self.env.render()
        time.sleep(self.RENDER_INTERVAL)

    def mapping_action_discrete2continues(self, action):
        return self.actions[action]

    def mapping_action(self, action):
        """
        入力される行動の範囲が-1から＋1であることを仮定
        """
        assert (action >= -1) and (action <=
                                   1), 'expected actions are \"-1 to +1\". input actions are {}'.format(action)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (action + 1.0) * 0.5 * \
            (high - low)  # (-1,1) -> (low,high)
        action = np.clip(action, low, high)  # (-X,+Y) -> (low,high)
        return action

    def __del__(self):
        self.env.close()

    def user_direct_set_domain_parameters(self, domain_info, type='set_split2'):
        self.env.domainInfo.set_parameters(domain_info, type=type)

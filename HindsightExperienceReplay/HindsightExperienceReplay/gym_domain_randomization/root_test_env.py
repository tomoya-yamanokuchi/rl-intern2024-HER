import sys

from .UserDefinedSettings import UserDefinedSettings
from .Environment.EnvironmentFactory import EnvironmentFactory


def test():
    userDefinedSettings = UserDefinedSettings()
    environmentFactory = EnvironmentFactory(userDefinedSettings)
    domain_num = None
    env = environmentFactory.generate(domain_num=domain_num)

    for episode_num in range(5):

        # domain randomization by uniform distribution
        env.env.domainInfo.set_parameters()

        state = env.reset()
        print('state', state)

        domain_parameter = env.env.domainInfo.get_domain_parameters()
        print('domain_parameter', domain_parameter)

        for step_num in range(env.MAX_EPISODE_LENGTH):
            env.render()
            action = env.random_action_sample()
            print('action', action)
            next_state, reward, done, domain_parameter = env.step(action)
            state = next_state
            print('state', state)
            if done:
                break


if __name__ == '__main__':
    test()

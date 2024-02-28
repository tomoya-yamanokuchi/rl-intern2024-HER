import numpy as np
import gym
import os
import sys

from mpi4py import MPI
import random
import torch

# from HindsightExperienceReplay.arguments import get_args
from HindsightExperienceReplay import get_args
from HindsightExperienceReplay import ddpg_agent
# from HindsightExperienceReplay import UserDefinedSettings
from HindsightExperienceReplay import UserDefinedSettingsFactory
from HindsightExperienceReplay import EnvironmentFactory



"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""


def get_env_params(env, env_name):
    obs = env.reset()
    # close the environment

    if env_name != 'Pendulum':
        params = {'obs': obs['observation'].shape[0],
                  'goal': obs['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  }
        params['max_timesteps'] = env._max_episode_steps
    else:
        params = {'obs': obs['observation'].shape[0],
                  'goal': obs['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.MAX_ACTION,
                  }
        params['max_timesteps'] = env.MAX_EPISODE_LENGTH

    return params


def launch(args):

    # create the ddpg_agent
    env_name = args.env_name
    # if env_name != 'Pendulum':
    if env_name == 'FetchPush-v1':
        env = gym.make(args.env_name)
    else:
        # userDefinedSettings = UserDefinedSettings()

        userDefinedSettings = UserDefinedSettingsFactory.generate(env_name=env_name)
        environmentFactory  = EnvironmentFactory(userDefinedSettings)
        domain_num = None
        env = environmentFactory.generate(domain_num=domain_num)

    # set random seeds for reproduce
    # env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env, env_name)
    # create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)


# mpirun -np 1 python -u train.py --env-name='Pendulum'  --cuda | tee reach.log

'''
tomoya-y:
参考リポジトリ : https://github.com/TianhongDai/hindsight-experience-replay
'''
# mpirun -np 1 python3.8 -u ./HindsightExperienceReplay/train.py --env-name='Pendulum'  --cuda | tee reach.log

# mpirun -np 1 python3.8 -u ./usecase/her/train.py --env-name='FetchPush-v1'  --cuda | tee reach.log
# mpirun -np 1 python3.8 -u ./usecase/her/train.py --env-name='RobelDClawCube'  --cuda | tee reach.log

import os
import torch
from HindsightExperienceReplay import actor
from HindsightExperienceReplay import get_args
import gym
import numpy as np

from HindsightExperienceReplay.gym_domain_randomization import UserDefinedSettings
from HindsightExperienceReplay import EnvironmentFactory
# process the inputs


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


if __name__ == '__main__':
    args = get_args()
    # load the model param
    # ------ モデルのロード部分 ------
    model_path = args.save_dir + args.env_name + "/" +  args.model + '/model.pt'
    # model_path = os.path.join(str(args.save_dir), str(args.env_name), str(args.model), '/model.pt')
    print("model_path = ", model_path)
    # import ipdb; ipdb.set_trace()
    # -----
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    if args.env_name != 'Pendulum':
        # import ipdb; ipdb.set_trace()
        env = gym.make(args.env_name)
    else:
        userDefinedSettings = UserDefinedSettings()
        environmentFactory = EnvironmentFactory(userDefinedSettings)
        domain_num = None
        env = environmentFactory.generate(domain_num=domain_num)
    # get the env param
    observation = env.reset()

    '''
    とりま
    '''
    env.MAX_EPISODE_LENGTH = 100

    # get the environment params
    if args.env_name != 'Pendulum':
        env_params = {'obs': observation['observation'].shape[0],
                      'goal': observation['desired_goal'].shape[0],
                      'action': env.action_space.shape[0],
                      'action_max': env.action_space.high[0],
                      }
        env_params['max_timesteps'] = env._max_episode_steps
    else:
        env_params = {'obs': observation['observation'].shape[0],
                      'goal': observation['desired_goal'].shape[0],
                      'action': env.action_space.shape[0],
                      'action_max': env.MAX_ACTION,
                      }
        env_params['max_timesteps'] = env.MAX_EPISODE_LENGTH
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env.MAX_EPISODE_LENGTH):
            env.render()

            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))

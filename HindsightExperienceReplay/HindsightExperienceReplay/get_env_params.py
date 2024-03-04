from HindsightExperienceReplay.gym_domain_randomization.Environment.RobelDClawCube import RobelDClawCube as Env



def get_env_params(env: Env, env_name: str):
    obs = env.reset()
    # close the environment

    if env_name == 'FetchPush-v1':
        params = {
            'obs'       : obs['observation'].shape[0],
            'goal'      : obs['desired_goal'].shape[0],
            'action'    : env.action_space.shape[0],
            'action_max': env.action_space.high[0],
        }
        params['max_timesteps'] = env._max_episode_steps

    else:
        params = {
            'obs'       : obs['observation'].shape[0],
            'goal'      : obs['desired_goal'].shape[0],
            'action'    : env.action_space.shape[0],
            'action_max': env.MAX_ACTION,
        }
        params['max_timesteps'] = env.MAX_EPISODE_LENGTH
        import ipdb; ipdb.set_trace()

    return params

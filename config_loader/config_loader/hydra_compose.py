from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


def hydra_compose(config_path, config_name, env_name:str = None):
    GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    if env_name is None    : return compose(config_name=config_name)
    print(f"defaults.env={env_name}")
    if env_name is not None: return compose(config_name=config_name, overrides=[f"env={env_name}"])
    raise NotImplementedError()

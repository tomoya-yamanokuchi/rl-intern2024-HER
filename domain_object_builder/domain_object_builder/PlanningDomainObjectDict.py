from typing import TypedDict
from robel_dclaw_env.domain.environment.instance.simulation.base_environment import AbstractEnvironment
from omegaconf import DictConfig
from icem_torch.control_adaptor import AbstractControlAdaptor


class PlanningDomainObjectDict(TypedDict):
    env_object      : AbstractEnvironment
    config_env      : DictConfig
    init_state      : dict
    TaskSpaceDiff   : object
    save_dir        : str
    control_adaptor : AbstractControlAdaptor

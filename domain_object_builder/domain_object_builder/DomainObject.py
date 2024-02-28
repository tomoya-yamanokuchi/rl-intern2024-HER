from omegaconf import DictConfig
from config_loader import ConfigLoader


class DomainObject:
    def __init__(self):
        self.config_icem       = None
        self.config_datamodule = None
        self.xml_str           = None

    def set_config_loader(self, configLoader: ConfigLoader):
        self.configLoader = configLoader

    def set_config_env(self, config_env: DictConfig):
        self.config_env = config_env

    def set_env_object(self, env_object):
        from robel_dclaw_env.domain.environment.instance.simulation.base_environment import AbstractEnvironment
        self.env_object : AbstractEnvironment = env_object

    def set_env(self, env):
        from robel_dclaw_env.domain.environment.instance.simulation.base_environment import AbstractEnvironment
        self.env : AbstractEnvironment = env

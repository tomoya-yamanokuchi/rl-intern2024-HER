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

    def set_config_xml_generation(self, config_xml_generation: DictConfig):
        self.config_xml_generation = config_xml_generation

    def set_env_object(self, env_object):
        from robel_dclaw_env.domain.environment.instance.simulation.base_environment import AbstractEnvironment
        self.env_object : AbstractEnvironment = env_object

    def set_original_xml_path(self, original_xml_path: str):
        self.original_xml_path = original_xml_path

    def set_env_init_state(self, init_state):
        self.init_state = init_state

    def set_env(self, env):
        from robel_dclaw_env.domain.environment.instance.simulation.cube.CubeSimulationEnvironment import CubeSimulationEnvironment
        self.env : CubeSimulationEnvironment = env

    def set_TaskSpaceValueObject(self, TaskSpaceValueObject):
        self.TaskSpaceValueObject = TaskSpaceValueObject

    def set_TaskSpaceDiffValueObject(self, TaskSpaceDiffValueObject):
        self.TaskSpaceDiffValueObject = TaskSpaceDiffValueObject

from abc import ABCMeta, abstractmethod
from .StateDict import StateDict
from omegaconf import DictConfig
from .render import RenderImageDict


class AbstractEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config):
        pass

    # @abstractmethod
    # def set_model_file(self, xml_str: str) -> None:
    #     pass

    @abstractmethod
    def set_xml_path(self, xml_path: str) -> None:
        pass


    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def reset(self, state: StateDict, verbose: bool=False):
        pass

    @abstractmethod
    def set_task_space_ctrl(self):
        pass

    @abstractmethod
    def set_joint_space_ctrl(self):
        pass

    @abstractmethod
    def get_state(self) -> StateDict:
        pass

    @abstractmethod
    def step(self, is_view: bool = False):
        pass

    @abstractmethod
    def render(self) -> RenderImageDict:
        pass

    @abstractmethod
    def view(self):
        pass


from typing import TypedDict
from mujoco_py import MjSim
from omegaconf import DictConfig


class RenderingManagerParamDIct(TypedDict):
    sim            : MjSim
    canonical_rgb  : DictConfig
    config_render  : DictConfig
    config_texture : DictConfig
    config_camera  : DictConfig
    config_light   : DictConfig

from .texture.TextureFactory import TextureFactory
from .CanonicalRendering import CanonicalRendering
from .RandomizedRendering import RandomizedRendering
from .RenderImageDict import RenderImageDict
from .rendering import Rendering
from mujoco_py import MjSim
from omegaconf import DictConfig
from .camera import Camera, CameraArgDict
from .light import LightManager
from .RenderingManager import RenderingManager
from .RenderingManagerParamDIct import RenderingManagerParamDIct


class NullRenderingManager(RenderingManager):
    def __init__(self, paramDict: RenderingManagerParamDIct):
        self.sim            =  paramDict["sim"]
        self.canonical_rgb  =  paramDict["canonical_rgb"]
        self.config_render  =  paramDict["config_render"]
        self.config_texture =  paramDict["config_texture"]
        self.config_camera  =  paramDict["config_camera"]
        self.config_light   =  paramDict["config_light"]
        pass

    def render(self):
        pass

    def set_canonical_rgb(self, canonical_rgb):
        pass

    def register_new_randomized_texture_collection(self):
        pass

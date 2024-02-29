from .texture.TextureFactory import TextureFactory
from .CanonicalRendering import CanonicalRendering
from .RandomizedRendering import RandomizedRendering
from .RenderImageDict import RenderImageDict
from .rendering import Rendering
from mujoco_py import MjSim
from .camera import Camera, CameraArgDict
from .light import LightManager
from .RenderingManagerParamDIct import RenderingManagerParamDIct


class RenderingManager:
    def __init__(self, paramDict: RenderingManagerParamDIct):
        sim            =  paramDict["sim"]
        canonical_rgb  =  paramDict["canonical_rgb"]
        config_render  =  paramDict["config_render"]
        config_texture =  paramDict["config_texture"]
        config_camera  =  paramDict["config_camera"]
        config_light   =  paramDict["config_light"]
        # ----
        camera             = Camera(CameraArgDict(sim=sim, **config_camera))
        rendering          = Rendering(sim, config_render.width_capture,  config_render.height_capture, camera)
        texture            = TextureFactory.create(config_texture.randomization_mode)(sim, **config_texture)
        light_manager      = LightManager(sim, **config_light)
        # import ipdb; ipdb.set_trace()
        self.can_rendering = CanonicalRendering (rendering, texture, light_manager, config_texture.force_default_canonical, canonical_rgb)
        self.ran_rendering = RandomizedRendering(rendering, texture, light_manager, config_texture.force_default_canonical)

    def render(self) -> RenderImageDict:
        return RenderImageDict(
            # random_nonfix = self.ran_rendering.render(), # 順番もしかすると大事かも
            canonical     = self.can_rendering.render(),
        )

    def set_canonical_rgb(self, canonical_rgb):
        self.can_rendering.set_canonical_rgb(canonical_rgb)

    def register_new_randomized_texture_collection(self):
        self.ran_rendering.register_new_randomized_texture_collection()

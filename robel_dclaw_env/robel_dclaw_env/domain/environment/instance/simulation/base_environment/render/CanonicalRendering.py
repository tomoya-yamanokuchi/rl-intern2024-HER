from .texture.TextureFactory import TextureFactory, TexturePerStep
from .light import LightManager
from .rendering import Rendering


class CanonicalRendering:
    def __init__(self,
            rendering               : Rendering,
            texture                 : TexturePerStep,
            light_manager           : LightManager,
            force_default_canonical : bool,
            canonical_rgb           ,
            ):
        self.canonical_rgb           = canonical_rgb
        self.rendering               = rendering
        self.texture                 = texture
        self.light_manager           = light_manager
        self.force_default_canonical = force_default_canonical

    def render(self):
        self.texture.set_canonical_texture(self.canonical_rgb)
        if not self.force_default_canonical:
            self.texture.set_task_relevant_randomized_texture()
        # import ipdb; ipdb.set_trace()
        self.light_manager.set_light_params()
        # print(self.canonical_rgb)
        return self.rendering.render("canonical")

    def set_canonical_rgb(self, canonical_rgb):
        self.canonical_rgb = canonical_rgb

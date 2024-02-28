from .texture import TextureFactory, TexturePerStep
from .light import LightManager
from .rendering import Rendering


class RandomizedRendering:
    def __init__(self,
            rendering               : Rendering,
            texture                 : TexturePerStep,
            light_manager           : LightManager,
            force_default_canonical : bool,
        ):
        self.rendering               = rendering
        self.texture                 = texture
        self.light_manager           = light_manager
        self.force_default_canonical = force_default_canonical

    def render(self):
        self.texture.set_randomized_texture()
        self.light_manager.set_light_params()
        return self.rendering.render("random_nonfix")

    def register_new_randomized_texture_collection(self):
        self.texture.register_new_randomized_texture_collection(include_task_relevant_object=True)

from .TexturePerStep import TexturePerStep


class TextureFactory:
    @staticmethod
    def create(name):
        if name == "per_step" : return TexturePerStep

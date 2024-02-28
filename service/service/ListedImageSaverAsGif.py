import numpy as np
from .create_gif_from_Image_list import create_gif_from_Image_list
from .create_null_class import create_null_class


class ListedImageSaverAsGif:
    def __init__(self, null:bool=False):
        if null: self.img_saver = create_null_class(ConcreteListedImageSaverAsGif)
        else   : self.img_saver = ConcreteListedImageSaverAsGif()

    def initialize_container(self):
        self.img_saver.initialize_container()

    def append(self, img: np.ndarray):
        self.img_saver.append(img)

    def save_as_gif(self, save_path: str, duration:int=500, rgb:bool=True):
        self.img_saver.save_as_gif(save_path, duration, rgb)


class ConcreteListedImageSaverAsGif:
    def __init__(self):
        self.initialize_container()

    def initialize_container(self):
        self.x = []

    def append(self, img: np.ndarray):
        self.x.append(img)

    def save_as_gif(self, save_path: str, duration:int=500, rgb:bool=True):
        create_gif_from_Image_list(
            images   = self.x,
            fname    = save_path,
            duration = duration,
            rgb      = rgb,
        )

import cv2
import time
import mujoco_py
from ..render.RenderImageDict import RenderImageDict


class OnscreenViewer:
    def __init__(self, sim):
        self.__viewer = mujoco_py.MjViewer(sim); time.sleep(1)

    def view(self, pseud_image : RenderImageDict = None):
        self.__viewer.render()

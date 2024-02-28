import cv2
import time
import mujoco_py
import numpy as np
from ..render.RenderImageDict import RenderImageDict


class OffscreenViewer:
    def __init__(self, sim):
        self.__viewer    = mujoco_py.MjRenderContextOffscreen(sim, 0); time.sleep(1)
        self.window_name = 'offscree_viewer'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def view(self, image: RenderImageDict):
        if image is None: return
        # cv2.imshow(self.window_name, image['canonical'].channel_last)
        # cv2.imshow(self.window_name, image['random_nonfix'].channel_last)
        img_ran  = image['random_nonfix'].channel_last
        img_can  = image['canonical'].channel_last
        img_diff = (img_ran - img_can)
        images   = np.concatenate([img_ran, img_can, img_diff], axis=1)
        images   = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, images)
        cv2.waitKey(50)

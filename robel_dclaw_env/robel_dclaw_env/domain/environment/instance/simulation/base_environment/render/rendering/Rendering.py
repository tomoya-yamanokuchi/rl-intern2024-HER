from mujoco_py import MjSim
from .ImageObject import ImageObject
from .utility import flip_upside_down, reverse_channel
from torch_numpy_converter import to_tensor
from ..camera import Camera


class Rendering:
    def __init__(self, sim: MjSim, width_capture: int, height_capture: int, camera: Camera):
        self.sim            = sim
        self.width_capture  = width_capture
        self.height_capture = height_capture
        self.camera         = camera

    def render(self, camera_name: str):
        self.camera.set_camera_posture(camera_name)
        img_rgb = self.sim.render(
            width       = self.width_capture,
            height      = self.height_capture,
            camera_name = camera_name,
            depth       = False
        ) # -> (w, h, c), dtype=uint8 : [0, 255]
        img_rgb        = flip_upside_down(img_rgb)
        # img_bgr        = reverse_channel(img_rgb) # なんでわざわざBGRに変換してるのかな？？(次のデータセット収集のタイミングで直してほしい)
        # return ImageObject(img_bgr)
        return ImageObject(img_rgb)


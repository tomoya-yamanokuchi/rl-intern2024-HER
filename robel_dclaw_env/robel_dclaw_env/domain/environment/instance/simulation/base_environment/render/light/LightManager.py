from .Light import Light
from mujoco_py import MjSim

class LightManager:
    def __init__(self, sim: MjSim, light1: int, light2: int):
        self.light      = Light(sim, light1, light2)
        self.ambient    = 0 # ライトの色味の変具合
        self.shadowsize = 0 # canonicalとrandomizedの色情報を一貫させるため

    def set_light_params(self):
        self.light.set_light_ambient(self.ambient)
        self.light.set_light_castshadow(self.shadowsize)
        self.light.set_light_on()

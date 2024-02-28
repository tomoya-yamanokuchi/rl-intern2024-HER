from .ViewerFactory import ViewerFactory


class ViewerManager:
    def __init__(self, sim, config):
        self.sim    = sim
        self.config = config
        self.viewer = None

    def __create_viewer(self):
        self.viewer = ViewerFactory().create(self.config.viewer.is_Offscreen)(self.sim)

    def view(self, render=None):
        if self.viewer is None:
            self.__create_viewer()
        self.viewer.view(render)

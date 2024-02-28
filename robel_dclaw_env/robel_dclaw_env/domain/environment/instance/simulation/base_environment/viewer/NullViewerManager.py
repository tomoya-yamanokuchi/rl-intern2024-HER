from .ViewerFactory import ViewerFactory
from .ViewerManager import ViewerFactory


class NullViewerManager(ViewerFactory):
    def __init__(self, sim, config):
        pass

    def __create_viewer(self):
        pass

    def view(self, render=None):
        pass

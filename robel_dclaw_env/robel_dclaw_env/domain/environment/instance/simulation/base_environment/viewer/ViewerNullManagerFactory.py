from .ViewerManager import ViewerManager
from .NullViewerManager import NullViewerManager


class ViewerNullManagerFactory:
    @staticmethod
    def create(use_render: bool, sim, config):
        if use_render: return ViewerManager(sim, config)
        else         : return NullViewerManager(sim, config)

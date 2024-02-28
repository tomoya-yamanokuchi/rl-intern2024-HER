from .RenderingManager import RenderingManager
from .NullRenderingManager import NullRenderingManager
from .RenderingManagerParamDIct import RenderingManagerParamDIct


class RenderingNullObjectManagerFactory:
    @staticmethod
    def create(use_render: bool, paramDict: RenderingManagerParamDIct):
        if use_render : return RenderingManager(paramDict)
        else          : return NullRenderingManager(paramDict)

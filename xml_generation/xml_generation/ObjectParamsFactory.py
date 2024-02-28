from .object_parameter import RandomPushingObjectParamDict
from .object_parameter import CylinderPushingObjectParamDict
from .object_parameter import RectangularPushingObjectParamDict

from typing import TypedDict


class ObjectParamsFactory:
    @staticmethod
    def create(object_type: str) -> TypedDict:
        if "random"      in object_type: return RandomPushingObjectParamDict
        if "cylinder"    in object_type: return CylinderPushingObjectParamDict
        if "rectangular" in object_type: return RectangularPushingObjectParamDict
        raise NotImplementedError()

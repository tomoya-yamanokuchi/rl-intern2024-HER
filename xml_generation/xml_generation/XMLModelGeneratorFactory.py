from .ValveXMLModelGenerator import ValveXMLModelGenerator
from .PushingObjectXMLModelGenerator import PushingObjectXMLModelGenerator
# from .object_parameter import ObjectParamDict
from .AbstractXMLModelGenerator import AbstractXMLModelGenerator
from .CylinderPushingObjectXMLModelGenerator import CylinderPushingObjectXMLModelGenerator
from .RectangularPushingObjectXMLModelGenerator import RectangularPushingObjectXMLModelGenerator
from typing import TypedDict


class XMLModelGeneratorFactory:
    @staticmethod
    def create(object_type: str, object_params: TypedDict, save_dir: str) -> AbstractXMLModelGenerator:
        if "random"      in object_type: return PushingObjectXMLModelGenerator(object_params, save_dir)
        if "cylinder"    in object_type: return CylinderPushingObjectXMLModelGenerator(object_params, save_dir)
        if "rectangular" in object_type: return RectangularPushingObjectXMLModelGenerator(object_params, save_dir)
        raise NotImplementedError()

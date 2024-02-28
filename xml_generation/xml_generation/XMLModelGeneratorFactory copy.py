from .ValveXMLModelGenerator import ValveXMLModelGenerator
from .PushingObjectXMLModelGenerator import PushingObjectXMLModelGenerator
from .object_parameter import ObjectParamDict
from .AbstractXMLModelGenerator import AbstractXMLModelGenerator


class XMLModelGeneratorFactory:
    @staticmethod
    def create(env_name: str, object_params: ObjectParamDict, save_dir: str) -> AbstractXMLModelGenerator:
        # if "valve"   in env_name: return ValveXMLModelGenerator(object_params, save_dir)

        if "pushing" in env_name: return PushingObjectXMLModelGenerator(object_params, save_dir)
        if "cylinder" in env

        raise NotImplementedError()

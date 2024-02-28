import os
import robel_dclaw_env
import xml.etree.ElementTree as ET
from omegaconf import DictConfig
from .BaseEnvironment import BaseEnvironment


class ModelLoader:
    def __init__(self, base_env: BaseEnvironment, config: DictConfig):
        self.base_env = base_env
        self.config   = config

    def _modify_xml(self, xml_path: str, new_file_path: str) -> str:
        tree         = ET.parse(xml_path)
        root         = tree.getroot()
        object1_body = root.find(".//body[@name='object1']")
        include_tag  = object1_body.find('.//include')
        include_tag.set('file', new_file_path)
        return ET.tostring(root, encoding='unicode')

    def load(self, model_path: str = None):
        if model_path is not None:
            return self.base_env.load_model(model_path)
        # ---
        absolute_path = os.path.abspath(robel_dclaw_env.__file__)
        parent_dir    = os.path.dirname(absolute_path)
        model_path    = os.path.join(parent_dir, "domain", "environment", "model", self.config.model_file)
        model         = self.base_env.load_model(model_path)
        return model

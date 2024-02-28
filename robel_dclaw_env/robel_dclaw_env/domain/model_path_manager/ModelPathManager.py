import os
import xml.etree.ElementTree as ET
from omegaconf import DictConfig
from pprint import pprint
import robel_dclaw_env
from .XMLModifier import XMLModifier

# from .copy_with_relative_paths import copy_with_relative_paths
from .ModelLoader import ModelLoader


class ModelPathManager:
    def __init__(self):
        self.xml_modifier = XMLModifier()
        self.model_loader = ModelLoader()

    def get_default_model_path(self, config_env: DictConfig):
        xml_path = os.path.join(config_env.env.model_dir, config_env.env.model_file)

        temp_xml_path = self.model_loader.load_model_from_path_with_includes(
            original_xml_path = xml_path,
        )

        # 一時ファイルの内容を表示
        with open(temp_xml_path, 'r') as file:
            print(file.read())

        # import ipdb; ipdb.set_trace()
        return temp_xml_path


    def get_custom_model_path(self, config_env: DictConfig, config_dataset: DictConfig):
        xml_path    = os.path.join(config_env.env.model_dir, config_env.env.model_file)
        object_path = os.path.join(
            *config_dataset.values(),
            "object_current_{}.xml".format(config_dataset.dataset_name),
        )
        modified_xml_str1 = self._modify_object_path(xml_path, object_path)

        import ipdb; ipdb.set_trace()
        modified_xml_str2 = self._convert_relative_paths_to_absolute(
            xml_string = modified_xml_str1,
            base_path  = os.path.abspath(robel_dclaw_env.__file__)
        )
        return modified_xml_str2



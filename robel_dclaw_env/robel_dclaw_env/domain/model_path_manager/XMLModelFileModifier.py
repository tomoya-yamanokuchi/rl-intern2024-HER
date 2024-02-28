import os
from copy import deepcopy
from lxml import etree
import tempfile
import mujoco_py
from omegaconf import DictConfig


class XMLModelFileModifier:
    def create_temporal_file(self, original_file_path: str):
        self.tree      = etree.parse(original_file_path)
        self.root      = self.tree.getroot()
        self.temp_file = tempfile.NamedTemporaryFile(
            dir    = os.path.dirname(original_file_path),
            delete = True,
            suffix = ".xml"
        )
        print(self.temp_file.name)
        return deepcopy(self.temp_file.name)

    def delete_temporal_file(self):
        self.temp_file.close()

    def get_new_include_path_from_config(self, config_object_model: DictConfig):
        '''
        ・動的にオブジェクトごとのxmlモデルファイルを作成している
          わけではなく, 予めenv_dataで作成されてある各データセットの
          直下に配置されているxmlモデルを読みに行く
        '''
        return os.path.join(
            *config_object_model.values(),
            "object_current_{}.xml".format(config_object_model.dataset_name)
        )

    def modify_and_save_xml(self, new_include_path: str):
        # 指定されたbodyタグを見つけてinclude fileのパスを変更
        base_path                 = os.path.dirname(self.temp_file.name)
        relative_new_include_path = os.path.relpath(new_include_path, start=base_path)
        for body in self.root.xpath("//body[@name='object1']"):
            for include in body.xpath("./include"):
                include.attrib['file'] = relative_new_include_path
        self.tree.write(self.temp_file.name)
        # import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    original_file_path = '/nfs/monorepo_ral2023/robel_dclaw_env/robel_dclaw_env/domain/environment/model/dclaw3xh_pushing.xml'
    new_include_path   = '/nfs/monorepo_ral2023/env_dataset/ququ/object_current_ququ.xml'
    xml_modifier       = XMLModelFileModifier()
    new_model_file     = xml_modifier.create_temporal_file(original_file_path)
    xml_modifier.modify_and_save_xml(new_include_path)
    model              = mujoco_py.load_model_from_path(new_model_file)
    xml_modifier.delete_temporal_file()

import os
import shutil
import tempfile
from xml.etree import ElementTree
import mujoco_py


class ModelLoader:

    def load_model_from_path_with_includes(self, original_xml_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._copy_all_files_including_includes(original_xml_path, temp_dir)
            temp_xml_path = os.path.join(temp_dir, os.path.basename(original_xml_path))
            return self._load_model_from_xml_path(temp_xml_path)

    def _copy_all_files_including_includes(self, original_xml_path, destination_dir):
        original_dir = os.path.dirname(original_xml_path)
        shutil.copy2(original_xml_path, destination_dir)
        new_xml_path = os.path.join(destination_dir, os.path.basename(original_xml_path))

        tree = ElementTree.parse(new_xml_path)
        root = tree.getroot()

        # Handle included files and other file attributes.
        for include_elem in root.iter():
            if 'file' in include_elem.attrib:
                file_path = include_elem.attrib['file']
                original_file_path = os.path.normpath(os.path.join(original_dir, file_path))
                relative_path = os.path.relpath(original_file_path, original_dir)
                new_path = os.path.normpath(os.path.join(destination_dir, relative_path))

                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.copy2(original_file_path, new_path)
                include_elem.attrib['file'] = os.path.relpath(new_path, destination_dir) # Use relative path here

                # Recursive call for nested includes.
                self._copy_all_files_including_includes(original_file_path, os.path.dirname(new_path))

        tree.write(new_xml_path)

    def _load_model_from_xml_path(self, xml_path):
        return mujoco_py.load_model_from_path(xml_path)

import os
import shutil
import tempfile
from lxml import etree
import mujoco_py

def modify_xml_include(xml_path, temp_dir):
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    for include_tag in root.xpath('//include'):
        file_path = include_tag.attrib['file']
        if not os.path.isabs(file_path):
            original_file_path = os.path.join(os.path.dirname(xml_path), file_path)
            copy_file_structure(original_file_path, xml_path, temp_dir)
            relative_path = os.path.relpath(original_file_path, start=os.path.dirname(xml_path))
            include_tag.attrib['file'] = relative_path

    return tree

def copy_file_structure(original_file_path, original_dir, temp_dir):
    relative_path = os.path.relpath(original_file_path, start=os.path.dirname(original_dir))
    new_path = os.path.join(temp_dir, relative_path)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    shutil.copy2(original_file_path, new_path)

    # if the file is another xml, modify it too
    if original_file_path.endswith('.xml'):
        modify_and_copy_includes(original_file_path, temp_dir)

def modify_and_copy_includes(original_xml_path, temp_dir):
    tree = modify_xml_include(original_xml_path, temp_dir)
    relative_path = os.path.relpath(original_xml_path, start=os.path.dirname(original_xml_path))
    new_xml_path = os.path.join(temp_dir, relative_path)
    tree.write(new_xml_path)

def load_model_from_path_with_includes(original_xml_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        modify_and_copy_includes(original_xml_path, temp_dir)
        temp_xml_path = os.path.join(temp_dir, os.path.basename(original_xml_path))
        model = mujoco_py.load_model_from_path(temp_xml_path)
    return model

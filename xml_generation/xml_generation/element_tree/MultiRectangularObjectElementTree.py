from xml.etree import ElementTree
import numpy as np
from typing import List
from service import join_with_mkdir
import xml.dom.minidom as md
from .MultiRectangularBodyTree import MultiRectangularBodyTree
from .JointTree import JointTree
from .MujocoTree import MujocoTree
import tempfile
import os


class MultiRectangularObjectElementTree:
    def __init__(self):
        self.mujoco_tree = MujocoTree()
        self.root        = self.mujoco_tree.root
        self.num_block   = 4
        self.temp_file   = None

    def add_joint_tree(self):
        joint_tree = JointTree()
        self.root.insert(1, joint_tree.root)

    def add_body_tree(self, type: str, size: str, mass: list, friction: List[float]):
        assert len(mass) == self.num_block
        x_half_size, y_half_size, z_half_size = size.split(" ")
        width_per_block = (float(y_half_size) / self.num_block)
        # bias_position   = (width_per_block * 2)
        size_per_block  = "{} {} {}".format(x_half_size, width_per_block, z_half_size)
        # import ipdb; ipdb.set_trace()
        body_tree = MultiRectangularBodyTree(type=type)
        body_tree.add_geometry(pos="0 -{} 0".format(width_per_block*3), size=size_per_block, mass=mass[0], friction=friction, id=0)
        body_tree.add_geometry(pos="0 -{} 0".format(width_per_block*1), size=size_per_block, mass=mass[1], friction=friction, id=1)
        body_tree.add_geometry(pos="0  {} 0".format(width_per_block*1), size=size_per_block, mass=mass[2], friction=friction, id=2)
        body_tree.add_geometry(pos="0  {} 0".format(width_per_block*3), size=size_per_block, mass=mass[3], friction=friction, id=3)
        self.root.insert(1, body_tree.root)

    def save_xml(self, save_path : str = None):
        if save_path is None:
            save_dir  = '/nfs/monorepo_ral2023/robel_dclaw_env/robel_dclaw_env/domain/environment/model/pushing_object'
            save_path = join_with_mkdir(save_dir, "object_current.xml", is_end_file=True)
        document  = md.parseString(ElementTree.tostring(self.root, 'utf-8'))
        if ".xml" not in save_path:
            save_path = save_path + ".xml"
        with open(save_path, "w") as file:
            document.writexml(file, encoding='utf-8', newl='\n', indent='', addindent="\t")

    def save_xml_as_temporal_file(self, save_dir : str = None):
        # -----
        document  = md.parseString(ElementTree.tostring(self.root, 'utf-8'))
        # ---
        self.temp_file = tempfile.NamedTemporaryFile(
            dir    = os.path.dirname(save_dir),
            delete = True,
            suffix = ".xml"
        )
        # ---
        with open(self.temp_file.name, "w") as file:
            document.writexml(file, encoding='utf-8', newl='\n', indent='', addindent="\t")
        # import ipdb; ipdb.set_trace()

    def delete_temporal_xml_file(self):
        # -----
        self.temp_file.close()
        print("** delete_temporal_xml_file **")

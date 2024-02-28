from xml.etree import ElementTree
import numpy as np
from typing import List
from service import join_with_mkdir
import xml.dom.minidom as md
# from .BodyTree import BodyTree
from .CylinderBodyTree import CylinderBodyTree
from .JointTree import JointTree
from .MujocoTree import MujocoTree


class CylinderObjectElementTree:
    def __init__(self):
        self.mujoco_tree = MujocoTree()
        self.root = self.mujoco_tree.root

    def add_joint_tree(self):
        joint_tree = JointTree()
        self.root.insert(1, joint_tree.root)

    def add_body_tree(self, size: str, mass: float, friction: List[float]):
        # assert len(xy_pos.shape) == 1
        # dim = xy_pos.shape
        # assert dim == 2
        body_tree = CylinderBodyTree()
        pos       = "0 0 0"
        body_tree.add_geometry(pos=pos, size=size, mass=mass, friction=friction)
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


if __name__ == '__main__':

    xy_position = np.random.randn(10, 2)*0.01
    mass        = 0.05

    # import ipdb; ipdb.set_trace()
    pusing_etree = PushingObjectElementTree()
    pusing_etree.add_joint_tree()
    pusing_etree.add_body_tree(xy_position, mass)
    pusing_etree.save_xml()

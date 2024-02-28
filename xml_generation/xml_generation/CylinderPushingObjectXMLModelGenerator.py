import os
from .CenterAlignedConvexHullGenerator import CenterAlignedConvexHullGenerator
from .InsidePointsNormalizer import InsidePointsNormalizer
from .object_parameter import ObjectMass, ObjectFriction, CylinderPushingObjectParamDict
from .element_tree import CylinderObjectElementTree
from .PlotConvexManager import PlotConvexManager
from .AbstractXMLModelGenerator import AbstractXMLModelGenerator


class CylinderPushingObjectXMLModelGenerator(AbstractXMLModelGenerator):
    def __init__(self, object_params: CylinderPushingObjectParamDict, save_dir: str):
        self.save_dir    = save_dir
        self.radius      = object_params['radius']
        self.half_length = object_params['half_length']
        # ---
        self.object_mass     = ObjectMass(object_params['mass'])
        self.object_friction = ObjectFriction(
            sliding_friction   = object_params['sliding_friction'],
            torsional_friction = object_params['torsional_friction'],
            rolling_friction   = object_params['rolling_friction'],
        )

    def generate(self,
            fname : str = "object_temp",
        ) -> None :
        # ---
        pusing_etree = CylinderObjectElementTree()
        pusing_etree.add_joint_tree()
        pusing_etree.add_body_tree(
            size     = "{} {}".format(self.radius, self.half_length),
            mass     = self.object_mass.unit_inside_cylinder_mass(num_inside_cylinder=1),
            friction = self.object_friction.unit_inside_cylinder_mass(num_inside_cylinder=1),
        )
        pusing_etree.save_xml(save_path=os.path.join(self.save_dir, fname))
        # import ipdb; ipdb.set_trace()





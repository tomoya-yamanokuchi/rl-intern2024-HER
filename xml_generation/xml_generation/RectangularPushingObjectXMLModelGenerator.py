import os
from .CenterAlignedConvexHullGenerator import CenterAlignedConvexHullGenerator
from .InsidePointsNormalizer import InsidePointsNormalizer
from .object_parameter import ObjectMass, ObjectFriction, RectangularPushingObjectParamDict
from .element_tree import MultiRectangularObjectElementTree
from .PlotConvexManager import PlotConvexManager
from .AbstractXMLModelGenerator import AbstractXMLModelGenerator


class RectangularPushingObjectXMLModelGenerator(AbstractXMLModelGenerator):
    def __init__(self, object_params: RectangularPushingObjectParamDict, save_dir: str):
        self.save_dir    = save_dir
        self.x_half_size = object_params['x_half_size']
        self.y_half_size = object_params['y_half_size']
        self.z_half_size = object_params['z_half_size']
        # ---
        self.type            = "box"
        # self.object_mass     = ObjectMass(object_params['mass'])
        self.mass : list     = object_params['mass']
        self.object_friction = ObjectFriction(
            sliding_friction   = object_params['sliding_friction'],
            torsional_friction = object_params['torsional_friction'],
            rolling_friction   = object_params['rolling_friction'],
        )
        self.pusing_etree = None

    def generate(self,
            fname : str = "object_temp",
        ) -> None :
        # ---
        pusing_etree = MultiRectangularObjectElementTree()
        pusing_etree.add_joint_tree()
        pusing_etree.add_body_tree(
            type     = self.type,
            size     = "{} {} {}".format(self.x_half_size, self.y_half_size, self.z_half_size),
            mass     = self.mass,
            friction = self.object_friction.unit_inside_cylinder_mass(num_inside_cylinder=1),
        )
        pusing_etree.save_xml(save_path=os.path.join(self.save_dir, fname))
        # import ipdb; ipdb.set_trace()


    def generate_as_temporal_file(self,
            sub_dirname = "pushing_object",
        ) -> None :
        # ---
        self.pusing_etree = MultiRectangularObjectElementTree()
        self.pusing_etree.add_joint_tree()
        self.pusing_etree.add_body_tree(
            type     = self.type,
            size     = "{} {} {}".format(self.x_half_size, self.y_half_size, self.z_half_size),
            mass     = self.mass,
            friction = self.object_friction.unit_inside_cylinder_mass(num_inside_cylinder=1),
        )
        self.pusing_etree.save_xml_as_temporal_file(
            save_dir=os.path.join(self.save_dir, sub_dirname),
        )

    def get_temporal_file_name(self):
        return self.pusing_etree.temp_file.name

    def delete_temporal_xml_file(self):
        if self.pusing_etree is None:
            print("** nothing any temporal file **")
            return
        self.pusing_etree.delete_temporal_xml_file()


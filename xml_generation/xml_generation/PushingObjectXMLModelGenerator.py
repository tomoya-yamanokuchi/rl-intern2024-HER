import os
from .CenterAlignedConvexHullGenerator import CenterAlignedConvexHullGenerator
from .InsidePointsNormalizer import InsidePointsNormalizer
from .object_parameter import ObjectMass, ObjectFriction, RandomPushingObjectParamDict
from .element_tree import PushingObjectElementTree
from .PlotConvexManager import PlotConvexManager
from service import join_with_mkdir
from .AbstractXMLModelGenerator import AbstractXMLModelGenerator



class PushingObjectXMLModelGenerator(AbstractXMLModelGenerator):
    def __init__(self, object_params: RandomPushingObjectParamDict, save_dir: str):
        self.save_dir        = save_dir
        self.object_mass     = ObjectMass(object_params['mass'])
        self.object_friction = ObjectFriction(
            sliding_friction   = object_params['sliding_friction'],
            torsional_friction = object_params['torsional_friction'],
            rolling_friction   = object_params['rolling_friction'],
        )

    def generate(self,
            num_sample_convex       :   int = 7,
            num_points_1axis        :   int = 30,
            plane_horizon_min_target: float = -0.03,
            plane_horizon_max_target: float =  0.03,
            fname                   :   str = "object_temp"
        ) -> None :
        # grouped_save_dir      = join_with_mkdir(self.save_dir, fname, is_end_file=False)
        visualizer            = PlotConvexManager(save_dir=self.save_dir, fname=fname)
        convex_generator      = CenterAlignedConvexHullGenerator(num_sample_convex, visualizer)
        aligned_inside_points = convex_generator.generate(num_points_1axis)
        # ---
        normalizer = InsidePointsNormalizer(
            convex                   = convex_generator.convex,
            plane_horizon_min_target = plane_horizon_min_target,
            plane_horizon_max_target = plane_horizon_max_target
        )
        normalized_inside_points = normalizer.normalize(aligned_inside_points)
        num_inside_points        = normalized_inside_points.shape[0]
        # ---
        pusing_etree = PushingObjectElementTree()
        pusing_etree.add_joint_tree()
        pusing_etree.add_body_tree(
            xy_pos   = normalized_inside_points,
            mass     = self.object_mass.unit_inside_cylinder_mass(num_inside_points),
            friction = self.object_friction.unit_inside_cylinder_mass(num_inside_points),
        )
        # import ipdb; ipdb.set_trace()
        pusing_etree.save_xml(save_path=os.path.join(self.save_dir, fname))
        import ipdb; ipdb.set_trace()





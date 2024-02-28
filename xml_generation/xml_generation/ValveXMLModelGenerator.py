import os
from .CenterAlignedConvexHullGenerator import CenterAlignedConvexHullGenerator
from .InsidePointsNormalizer import InsidePointsNormalizer
from .object_parameter import ObjectMass, ObjectFriction # , ObjectParamDict
from .element_tree import PushingObjectElementTree
from .PlotConvexManager import PlotConvexManager
from service import join_with_mkdir
from .AbstractXMLModelGenerator import AbstractXMLModelGenerator
from typing import TypedDict

class ValveXMLModelGenerator(AbstractXMLModelGenerator):
    def __init__(self, object_params : TypedDict, save_dir: str):
        pass

    def generate(self,
            num_sample_convex       :   int = 7,
            num_points_1axis        :   int = 30,
            plane_horizon_min_target: float = -0.03,
            plane_horizon_max_target: float =  0.03,
            fname                   :   str = "object_temp"
        ) -> None :
        pass

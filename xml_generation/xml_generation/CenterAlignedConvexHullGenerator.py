from copy import deepcopy
import numpy as np
from .ConvexHull2D import ConvexHull2D
from .flattened_2d_meshgrid import flattened_2d_meshgrid
from .PlotConvexManager import PlotConvexManager


class CenterAlignedConvexHullGenerator:
    def __init__(self, num_sample_convex: int = 7, visualizer: PlotConvexManager = None) -> None:
        self.convex     = ConvexHull2D(num_sample_convex)
        self.visualizer = visualizer

    def generate(self, num_points_1axis: int = 30) -> np.ndarray:
        '''
         - num_points_1axis: 1辺あたりのcylinderの敷き詰める数
        '''
        whole_area_points       = flattened_2d_meshgrid(self.convex.min, self.convex.max, num_points_1axis)
        inside_points           = self.convex.get_inside_points(whole_area_points)
        aligned_convex          = self.__align_convex_center(inside_points)
        aligned_inside_points   = aligned_convex.get_inside_points(whole_area_points)
        # ---
        if self.visualizer is not None:
            self.visualizer.plot_convex_origin(self.convex, whole_area_points, inside_points)
            self.visualizer.plot_convex_aligned(aligned_convex, whole_area_points, aligned_inside_points)
        return aligned_inside_points

    def __align_convex_center(self, inside_points) -> ConvexHull2D:
        # --- 生成した凸包の中心が(0, 0)に来るようにする ---
        aligned_convex = deepcopy(self.convex)
        aligned_convex.hull.points[:, 0] += (inside_points[:, 0].mean())*(-1)
        aligned_convex.hull.points[:, 1] += (inside_points[:, 1].mean())*(-1)
        return aligned_convex

import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from .sample_points_in_unit_circle import sample_points_in_unit_circle


class ConvexHull2D:
    def __init__(self, num_sample: int=7, sampling_min: float=-1.0, sampling_max: float=1.0):
        self.num_sample = num_sample
        self.min        = sampling_min
        self.max        = sampling_max
        self.jitter     = 1e-16
        self.hull       = self.__generate_and_compute_convex_hull()

    def __generate_and_compute_convex_hull(self):
        random_2d_points = sample_points_in_unit_circle(self.num_sample)
        assert random_2d_points.min() > (self.min - self.jitter)
        assert random_2d_points.max() < (self.max + self.jitter)
        return ConvexHull(random_2d_points)

    def get_inside_points(self, points: np.ndarray):
        assert len(points.shape) == 2 # (num_data, dim)
        assert points.shape[-1] == 2 # dim == 2 (2次元平面)
        # ---
        hull_path     = Path(self.hull.points[self.hull.vertices])
        num_data      = points.shape[0]
        inside_points = []
        for i in range(num_data):
            if not hull_path.contains_point(points[i]): continue
            inside_points.append(points[i])
        inside_points = np.stack(inside_points)
        return inside_points

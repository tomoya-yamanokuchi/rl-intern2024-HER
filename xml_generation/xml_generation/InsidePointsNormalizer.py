import numpy as np
from .ConvexHull2D import ConvexHull2D
from service import normalize


class InsidePointsNormalizer:
    def __init__(self,
            convex                  : ConvexHull2D,
            plane_horizon_min_target: float = -0.03,
            plane_horizon_max_target: float =  0.03,
            ):
        self.convex                   = convex
        self.plane_horizon_min_target = plane_horizon_min_target
        self.plane_horizon_max_target = plane_horizon_max_target

    def normalize(self, aligned_inside_points) -> np.ndarray:
        return normalize(
            x     = aligned_inside_points,
            x_min = self.convex.min,
            x_max = self.convex.max,
            m     = self.plane_horizon_min_target,
            M     = self.plane_horizon_max_target,
        )

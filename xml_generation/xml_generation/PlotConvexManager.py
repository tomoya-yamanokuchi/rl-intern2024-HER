import os
from .PlotConvexHull import PlotConvexHull


class PlotConvexManager:
    def __init__(self, save_dir: str, fname: str):
        self.save_dir = save_dir
        self.fname    = fname

    def plot_convex_origin(self, convex, whole_area_points, inside_points):
        self.__plot_convex(convex, whole_area_points, inside_points,
            os.path.join(self.save_dir, "{}_origin.png".format(self.fname)))

    def plot_convex_aligned(self, convex, whole_area_points, inside_points):
        self.__plot_convex(convex, whole_area_points, inside_points,
            os.path.join(self.save_dir, "{}_aligned.png".format(self.fname)))

    def __plot_convex(self, convex, whole_area_points, inside_points, save_path):
        plot_convex = PlotConvexHull(convex)
        plot_convex.plot(whole_area_points, inside_points, save_path)

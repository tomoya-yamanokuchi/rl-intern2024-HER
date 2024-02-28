import numpy as np
import matplotlib.pyplot as plt


def plot_convex_hull(hull):
    plt.figure()

    # ポイントのプロット
    plt.plot(hull.points[:,0], hull.points[:,1], 'o')

    # 凸包を形成する点の順序を持つリスト
    hull_indices = np.append(hull.vertices, hull.vertices[0])

    # 凸包の線のプロット
    plt.plot(hull.points[hull_indices,0], hull.points[hull_indices,1], 'r-')

    plt.show()

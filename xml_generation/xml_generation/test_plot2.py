import numpy as np
import matplotlib.pyplot as plt



def sample_points_in_unit_circle(num_sample):
    np.random.seed()
    r      = np.random.rand(num_sample)               # range: [0, 1)
    theta  = 2*np.pi * np.random.rand(num_sample) - np.pi    # range: [-pi, pi)
    # import ipdb; ipdb.set_trace()
    points = np.array([r*np.cos(theta),r*np.sin(theta)]).T # range: (-1, 1)
    return points

# 関数を呼び出して点を生成します
points = _random_2d_points_generation(1000)

# プロットします
plt.figure(figsize=(5, 5))
plt.scatter(points[:, 0], points[:, 1], alpha=0.6)
plt.title("Randomly Generated Points within a Unit Circle")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

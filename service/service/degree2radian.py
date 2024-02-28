import torch
import math


def degree2radian(euler_angles_degrees: torch.Tensor):
    return (euler_angles_degrees * math.pi / 180.0)

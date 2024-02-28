import torch
import numpy as np
from torch_numpy_converter import to_tensor


def to_tensor_with_type_check(x):
    if   isinstance(x, torch.Tensor): x_tensor = x
    elif isinstance(x, np.ndarray)  : x_tensor = to_tensor(x.copy())
    else                                : raise NotImplementedError()
    # ----
    return x_tensor

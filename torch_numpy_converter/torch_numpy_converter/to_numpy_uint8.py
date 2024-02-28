from torch import Tensor
import numpy as np

def to_numpy_uint8(x: Tensor):
    return x.detach().to('cpu').numpy().astype(np.uint8)




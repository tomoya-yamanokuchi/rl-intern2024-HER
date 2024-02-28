import torch
import numpy as np

def to_tensor(x):
    # import ipdb; ipdb.set_trace()
    if isinstance(x, np.ndarray)  : return torch.from_numpy(x).contiguous().type(torch.cuda.FloatTensor)
    if isinstance(x, torch.Tensor): return x
    # import ipdb; ipdb.set_trace()
    raise NotImplementedError()

def to_tensor_double(x):
    return torch.from_numpy(x).contiguous().type(torch.cuda.DoubleTensor)

def to_tensor_without_cuda(x):
    return torch.from_numpy(x).contiguous()

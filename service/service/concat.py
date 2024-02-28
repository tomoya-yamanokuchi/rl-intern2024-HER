import torch


def concat(w_current, w_new, dim=0):
    if w_current is None: return w_new
    return torch.cat([w_current, w_new], dim=dim)

from torch import Tensor
from .concat import concat


class DictTensorContainer:
    def __init__(self):
        self.container = {}

    def concatenate(self, x : Tensor, key: str, dim: int):
        current_value       = self.container.get(key, None) # キーが存在しなければNoneを返す
        self.container[key] = concat(current_value, x, dim=dim)

    def get_item(self, key: str):
        return self.container[key]

from .StateDict import StateDict
from copy import deepcopy
import numpy as np


class State:
    def __init__(self, state_dict: StateDict):
        self.state_dict = state_dict

    def get_as_object(self):
        return deepcopy(self.state_dict)

    def get_as_ndarray(self):
        return {key: np.array(value_object.value) for key, value_object in self.state_dict.items()}

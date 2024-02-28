from collections import defaultdict
from typing import List, Dict, Any
# import numpy as np
import torch


def stack_by_key_torch(dict_list: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
    stacked_dict = {}
    for key, value_list in dict_list.items():
        for val in value_list:
            # import ipdb; ipdb.set_trace()
            print(key, val.shape)
        stacked_dict[key] = torch.stack(value_list, dim=0)
    # return {key: np.stack(value_list, axis=0) for key, value_list in dict_list.items()}
    return stacked_dict

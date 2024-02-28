from collections import defaultdict
from typing import List, Dict, Any
import numpy as np



def stack_by_key(dict_list: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
    stacked_dict = {}
    for key, value_list in dict_list.items():
        for val in value_list:
            # import ipdb; ipdb.set_trace()
            print(key, val.shape)
        stacked_dict[key] = np.stack(value_list, axis=0)
    # return {key: np.stack(value_list, axis=0) for key, value_list in dict_list.items()}
    return stacked_dict

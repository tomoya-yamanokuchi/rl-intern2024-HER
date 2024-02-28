import numpy as np
import torch
from robel_dclaw_env.domain.forward_model_multiprocessing.ForkedPdb import ForkedPdb
from pprint import pprint
from .create_empty_list_dict import create_empty_list_dict

irrelevant_key = ["time", "act", "udd_state"]


class CollectionAggregation:
    def aggregate(self, dataclass_list: list):
        keys = dataclass_list[0].keys()
        data = create_empty_list_dict(keys) # pprint(data)
        # -----
        for t in range(len(dataclass_list)):
            x = dataclass_list[t]
            for key in keys:
                # print("key = ", key)
                val = x[key].value
                val = val.squeeze() if type(val) == np.ndarray else val
                data[key].append(val)
        # -----
        for key in keys:
            if key in irrelevant_key: continue
            data[key] = torch.stack(data[key])
        # -----
        # ForkedPdb().set_trace()
        return data

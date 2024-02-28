from typing import List



def convert_target_to_string(taregt: List[float]):
    target_str_list = []
    for _target in taregt:
        tt = str(_target).replace(".", "p")
        target_str_list.append(tt)
    target_str = "_".join(target_str_list)
    target_str = "[{}]".format(target_str)
    # import ipdb; ipdb.set_trace()
    return target_str

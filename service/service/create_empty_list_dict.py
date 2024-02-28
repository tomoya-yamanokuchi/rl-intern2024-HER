from typing import List


def create_empty_list_dict(keys: List[str]):
    aggregation_results = {}
    for key in keys:
        aggregation_results[key] = []
    return aggregation_results

from typing import TypedDict, List, Any
import torch


def stack_tensors_in_dicts(dict_list: List[TypedDict], dim: int) -> TypedDict:
    # Initialize an empty dictionary to store the combined tensors
    combined_dict = {}

    # Check if the list is empty
    if not dict_list:
        return combined_dict

    # Iterate over each key in the first dictionary (assuming all dictionaries have the same structure)
    for key in dict_list[0].keys():
        # Combine the tensors for this key from all dictionaries in the list
        combined_dict[key] = torch.stack([d[key] for d in dict_list], dim=dim)

    return combined_dict


if __name__ == '__main__':
    # Example usage
    # Creating some mock data for demonstration purposes
    dict1 = {'a': torch.randn(5), 'b': torch.randn(5)}
    dict2 = {'a': torch.randn(5), 'b': torch.randn(5)}

    # Combine the dictionaries
    combined_dict = stack_tensors_in_dicts([dict1, dict2], dim=0)
    print(combined_dict["a"].shape)
    print(combined_dict["b"].shape)

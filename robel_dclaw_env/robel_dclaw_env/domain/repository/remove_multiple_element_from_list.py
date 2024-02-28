



def remove_multiple_element_from_list(in_list: dict, keys_to_remove: list) -> None:
    for key in keys_to_remove:
        if key in in_list:
            in_list.remove("banana")





def parameters_to_directory_name(params_dict):
    name = ""
    for key, val in params_dict.items():
        name += "[{}={}]".format(key, val)
        name += "-"
    return name

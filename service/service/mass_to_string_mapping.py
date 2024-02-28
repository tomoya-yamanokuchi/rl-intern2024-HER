


# def mass_to_string_mapping(mass: list):
#     assert len(mass) == 3
#     mass_str = str(mass).replace("[", "").replace("]", "").split(", ")
#     mass_str = "{:.0e}_{:.0e}_{:.0e}".format(float(mass_str[0]), float(mass_str[1]), float(mass_str[2]))
#     mass_str = mass_str.replace("-", "_")
#     return mass_str

def mass_to_string_mapping(mass: list):
    assert len(mass) == 4
    # mass_str = str(mass).replace("[", "").replace("]", "").split(", ")
    # mass_str = "{:.1e}_{:.1e}_{:.1e}_{:.1e}".format(float(mass_str[0]), float(mass_str[1]), float(mass_str[2]), float(mass_str[3]))

    # import ipdb; ipdb.set_trace()
    mass_str_formatted = ["{:.1e}".format(m).replace('.', 'p') for m in mass]
    mass_str_joined    = "_".join(mass_str_formatted)
    mass_str           = mass_str_joined.replace("-", "")
    return mass_str

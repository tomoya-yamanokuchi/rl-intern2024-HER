from socket import gethostname


def get_pc_name():
    hostname_dict = {
        "dl-box0"              : "dl-box",
        "G-Master-Spear-MELCO" : "remote_3090_1",
        "tycluster1"           : "remote_3090_2",
        "tsukumo3090ti"        : "melco_tsukumo3090ti",
    }
    name = gethostname()
    return hostname_dict[name]

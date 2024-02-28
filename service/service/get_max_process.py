import os


def get_max_process(num_input: int = None, verbose: bool=True):
    if num_input is None : max_process = int(os.cpu_count()/2)
    else                 : max_process = min(num_input, int(os.cpu_count()/2))
    # ----
    if verbose: print("\n [ os max_process = {} ] \n".format(max_process))
    return max_process


import time
import numpy as np


def wait_time_const(wait_time=5, verbose=False):
    if verbose: print("wait_time = {:.3f}".format(wait_time))
    time.sleep(wait_time)

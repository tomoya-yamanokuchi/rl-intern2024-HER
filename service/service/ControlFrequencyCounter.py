import os
import time
import numpy as np


class ControlFrequencyCounter:
    def __init__(self):
        self._time_hz    = None
        self._hz_history = []

    def update(self):
        time_current = time.time()
        if self._time_hz is not None:
            update_time  = (time_current - self._time_hz)
            hz           = (1. / update_time)
            self._hz_history.append(hz)
            # print("update_time = {}".format(update_time))
            # print(" ctrl frequency = {: .3f} [Hz]".format(hz))
        self._time_hz = time_current

    def save_average_hz(self, save_dir):
        average_hz = np.mean(self._hz_history)
        print("=============================================")
        print(" average ctrl frequency = {: .3f} [Hz]".format(average_hz))
        print("=============================================")
        np.save(
            file = os.path.join(save_dir, "average_ctrl_frequency.npy"),
            arr  = average_hz,
        )

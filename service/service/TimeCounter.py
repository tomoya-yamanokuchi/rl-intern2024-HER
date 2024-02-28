import time


class TimeCounter:
    def __init__(self):
        self.__reset()
        self.total_time = 0.0

    def __reset(self):
        self.time_start = None

    def start(self):
        assert self.time_start is None
        self.time_start = time.time()

    def stop(self):
        assert self.time_start is not None
        time_stop    = time.time()
        elapsed_time = time_stop - self.time_start
        self.__reset()
        self.total_time += elapsed_time
        return elapsed_time

    def get_total_time(self):
        return self.total_time

    def print_total_time(self):
        print("-------------------------------")
        print("     Total time = {} [sec] ".format(self.total_time))
        print("-------------------------------")

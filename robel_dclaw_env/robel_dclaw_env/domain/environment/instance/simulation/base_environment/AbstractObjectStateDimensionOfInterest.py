from abc import ABCMeta, abstractmethod


class AbstractObjectStateDimensionOfInterest(metaclass=ABCMeta):
    @abstractmethod
    def extract(self, qinfo):
        pass


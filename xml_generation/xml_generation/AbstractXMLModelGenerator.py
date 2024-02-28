from abc import ABC, abstractmethod
from typing import TypedDict


class AbstractXMLModelGenerator(ABC):
    @abstractmethod
    def __init__(self, object_params: TypedDict, save_dir: str) -> None:
        pass

    @abstractmethod
    def generate(self, fname, **kwargs) -> None :
        pass

    @abstractmethod
    def generate_as_temporal_file(self,
        ) -> None :
        pass

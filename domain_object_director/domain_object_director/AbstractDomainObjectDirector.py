from abc import ABC, abstractmethod
from domain_object_builder import DomainObjectBuilder, DomainObject, ModelSetParamsDict


class AbstractDomainObjectDirector(ABC):
    @abstractmethod
    def construct(builder: DomainObjectBuilder, metadata: ModelSetParamsDict, **kwargs) -> DomainObject:
        pass

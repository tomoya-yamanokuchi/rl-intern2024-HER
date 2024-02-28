from .KVAEDomainObjectDirector import KVAEDomainObjectDirector
from .ModelDomainObjectDirector import ModelDomainObjectDirector
from .ViRCCEDomainObjectDirector import ViRCCEDomainObjectDirector
from ..AbstractDomainObjectDirector import AbstractDomainObjectDirector


class DomainObjectDirectorFactory:
    @staticmethod
    def create(model_class: str) -> AbstractDomainObjectDirector:
        if "kvae"   in model_class : return KVAEDomainObjectDirector()
        if "cdsvae" in model_class : return ModelDomainObjectDirector()
        if "vircce" in model_class : return ViRCCEDomainObjectDirector()
        raise NotImplementedError()

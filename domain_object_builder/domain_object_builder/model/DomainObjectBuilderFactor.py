from .kvae import KVAEDomainObjectBuilder
from .ModelDomainObjectBuilder import ModelDomainObjectBuilder
from .ViRCCEDomainObjectBuilder import ViRCCEDomainObjectBuilder
from omegaconf import DictConfig


class DomainObjectBuilderFactor:
    @staticmethod
    def create(model_class: str, config: DictConfig) -> object:
        if "kvae"   in model_class : return KVAEDomainObjectBuilder(config)
        if "cdsvae" in model_class : return ModelDomainObjectBuilder(config)
        if "vircce" in model_class : return ViRCCEDomainObjectBuilder(config)
        raise NotImplementedError()

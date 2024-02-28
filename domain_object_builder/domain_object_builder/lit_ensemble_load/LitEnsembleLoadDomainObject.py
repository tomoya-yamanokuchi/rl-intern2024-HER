from torch import nn
from omegaconf import DictConfig


class LitEnsembleLoadDomainObject:
    def set_lit_model(self, lit_model):
        from cdsvae.domain.model import AbstractLitEnsemble
        self.lit_model : AbstractLitEnsemble = lit_model

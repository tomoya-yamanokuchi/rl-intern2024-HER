from torch import nn
from omegaconf import DictConfig


class LitModelLoadDomainObject:
    def set_lit_model(self, lit_model):
        from cdsvae.domain.model import AbstractLitModel
        self.lit_model : AbstractLitModel = lit_model

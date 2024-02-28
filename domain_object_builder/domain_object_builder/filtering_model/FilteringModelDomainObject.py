from torch import nn
from omegaconf import DictConfig


class FilteringModelDomainObject:
    # ---- config ----
    def set_config_loader(self, configLoader):
        from config_loader import ConfigLoader
        self.configLoader : ConfigLoader = configLoader

    def set_config(self, config: DictConfig):
        self.config = config

    def set_config(self, config: DictConfig):
        self.config = config

    # ---- After config ----
    def set_filteirng(self, filteirng):
        from cdsvae.domain.filtering import AbstractFiltering
        self.filteirng : AbstractFiltering = filteirng


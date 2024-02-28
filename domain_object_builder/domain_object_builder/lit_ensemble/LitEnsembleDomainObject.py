from torch import nn
from omegaconf import DictConfig


class LitEnsembleDomainObject:
    # ---- config ----
    def set_config_loader(self, configLoader):
        from config_loader import ConfigLoader
        self.configLoader : ConfigLoader = configLoader

    def set_config(self, config: DictConfig):
        self.config = config

    def set_config(self, config: DictConfig):
        self.config = config

    def set_model(self, model):
        from cdsvae.domain.model import AbstractEnsemble
        self.model : AbstractEnsemble = model

    def set_training_loss(self, training_loss):
        from cdsvae.domain.loss.ensemble import TrainingLoss
        self.training_loss : TrainingLoss = training_loss

    def set_validation_loss(self, validation_loss):
        from cdsvae.domain.loss.ensemble import ValidationLoss
        self.validation_loss : ValidationLoss = validation_loss

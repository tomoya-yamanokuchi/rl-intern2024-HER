from torch import nn
from omegaconf import DictConfig


class LitModelDomainObject:
    # ---- config ----
    def set_config_loader(self, configLoader):
        from config_loader import ConfigLoader
        self.configLoader : ConfigLoader = configLoader

    def set_config(self, config: DictConfig):
        self.config = config

    def set_config(self, config: DictConfig):
        self.config = config

    def set_model(self, model):
        from cdsvae.domain.model import AbstractModel
        self.model : AbstractModel = model

    def set_image_logger(self, image_logger):
        from image_logger import ImageLogger
        self.image_logger : ImageLogger = image_logger

    # ---- loss ----
    def set_weight(self, weight : float):
        self.weight = weight

    def set_contrastive_mi_c(self, contrastive_mi_c: nn.Module):
        self.contrastive_mi_c = contrastive_mi_c

    def set_contrastive_mi_m(self, contrastive_mi_m: nn.Module):
        self.contrastive_mi_m = contrastive_mi_m

    def set_mutual_information(self, mutual_information: nn.Module):
        self.mutual_information = mutual_information

    def set_training_loss(self, training_loss):
        from cdsvae.domain.loss.model import TrainingLoss
        self.training_loss : TrainingLoss = training_loss

    def set_validation_loss(self, validation_loss):
        from cdsvae.domain.loss.model import ValidationLoss
        self.validation_loss : ValidationLoss = validation_loss

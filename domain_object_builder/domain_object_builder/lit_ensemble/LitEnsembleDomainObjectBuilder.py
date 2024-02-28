from omegaconf import DictConfig
from .LitEnsembleDomainObject import LitEnsembleDomainObject


class LitEnsembleDomainObjectBuilder:
    def __init__(self, model):
        self.domain_object = LitEnsembleDomainObject()
        self.domain_object.set_model(model)

    def build_config_loader(self):
        from config_loader import ConfigLoader
        self.domain_object.set_config_loader(ConfigLoader())

    def build_config(self):
        config = self.domain_object.configLoader.load_ensemble()
        self.domain_object.set_config(config=config)

    def build_training_loss(self):
        from cdsvae.domain.loss.ensemble.train import TrainingLoss
        from cdsvae.domain.loss.ensemble.utils import ConstructorParamsDict
        paramsDict = ConstructorParamsDict(
            model        = self.domain_object.model,
            beta         = self.domain_object.config.loss.beta,
            num_ensemble = self.domain_object.config.num_ensemble,
        )
        training_loss = TrainingLoss(paramsDict)
        self.domain_object.set_training_loss(training_loss)

    def build_validation_loss(self):
        from cdsvae.domain.loss.ensemble.valid import ValidationLoss
        from cdsvae.domain.loss.ensemble.utils import ConstructorParamsDict
        paramsDict = ConstructorParamsDict(
            model        = self.domain_object.model,
            beta         = self.domain_object.config.loss.beta,
            num_ensemble = self.domain_object.config.num_ensemble,
        )
        validation_loss = ValidationLoss(paramsDict)
        self.domain_object.set_validation_loss(validation_loss)

    def get_domain_object(self) -> LitEnsembleDomainObject:
        return self.domain_object

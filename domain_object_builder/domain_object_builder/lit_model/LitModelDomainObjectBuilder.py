from omegaconf import DictConfig
from .LitModelDomainObject import LitModelDomainObject


class LitModelDomainObjectBuilder:
    def __init__(self, model):
        self.domain_object = LitModelDomainObject()
        self.domain_object.set_model(model)

    def build_config_loader(self):
        from config_loader import ConfigLoader
        self.domain_object.set_config_loader(ConfigLoader())

    def build_config(self):
        config = self.domain_object.configLoader.load_model()
        self.domain_object.set_config(config=config)

    def build_image_logger(self):
        from image_logger import ImageLogger
        image_logger = ImageLogger(max_save_num=12, rgb=True) # DomainObjectBuilderの中のimage_loggerとは別物
        self.domain_object.set_image_logger(image_logger)

    # ---- loss ----
    def build_weight(self):
        self.domain_object.set_weight(weight=self.domain_object.config.model.loss.weight)

    def build_contrastive_mi_c(self):
        from cdsvae.domain.loss.contrastive_mutual_information import ContrastiveMutualInformationFactory
        contrastive_mi_c = ContrastiveMutualInformationFactory.create(
            **self.domain_object.config.model.loss.contrastive_mi.content,
            config_model = self.domain_object.config.model,
        )
        self.domain_object.set_contrastive_mi_c(contrastive_mi_c)

    def build_contrastive_mi_m(self):
        from cdsvae.domain.loss.contrastive_mutual_information import ContrastiveMutualInformationFactory
        contrastive_mi_m   = ContrastiveMutualInformationFactory.create(**self.domain_object.config.model.loss.contrastive_mi.motion )
        self.domain_object.set_contrastive_mi_m(contrastive_mi_m)

    def build_mutual_information(self):
        from cdsvae.domain.loss.mutual_information import MutualInformationFactory
        mutual_information = MutualInformationFactory().create(
            **self.domain_object.config.model.loss.mutual_information,
            num_train = self.domain_object.config.datamodule.num_data.num_train
        )
        self.domain_object.set_mutual_information(mutual_information)

    def build_training_loss(self, model_class: str):
        from cdsvae.domain.loss.model import TrainingModelLossFactory
        TrainingModelLossObject, ParamsDict = TrainingModelLossFactory.create(model_class=model_class)
        # import ipdb; ipdb.set_trace()
        paramsDict = ParamsDict(
            weight             = self.domain_object.config.model.loss.weight,
            contrastive_mi_c   = self.domain_object.contrastive_mi_c,
            contrastive_mi_m   = self.domain_object.contrastive_mi_m,
            mutual_information = self.domain_object.mutual_information,
            model              = self.domain_object.model,
            config_model       = self.domain_object.config.model,
        )
        training_loss = TrainingModelLossObject(paramsDict)
        self.domain_object.set_training_loss(training_loss)

    def build_validation_loss(self, model_class: str):
        from cdsvae.domain.loss.model import ValidationModelLossFactory
        ValidationModelLossObject, ParamsDict = ValidationModelLossFactory.create(model_class=model_class)
        paramsDict = ParamsDict(
            model              = self.domain_object.model,
        )
        validation_loss = ValidationModelLossObject(paramsDict)
        self.domain_object.set_validation_loss(validation_loss)

    def get_domain_object(self) -> LitModelDomainObject:
        return self.domain_object

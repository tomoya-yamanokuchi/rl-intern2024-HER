from omegaconf import DictConfig
from .FilteringModelDomainObject import FilteringModelDomainObject


class FilteringModelDomainObjectBuilder:
    def __init__(self):
        self.domain_object = FilteringModelDomainObject()

    def build_config_loader(self):
        from config_loader import ConfigLoader
        self.domain_object.set_config_loader(ConfigLoader())

    def build_config(self):
        config = self.domain_object.configLoader.load_model()
        self.domain_object.set_config(config=config)

    def build_filtering(self, model_domain_object):
        # from domain_object_builder.model import ModelDomainObjectBuilder
        # from domain_object_director.model import MoelDomainObjectDirector
        from cdsvae.domain.filtering import Filtering
        # ----
        # model_builder       = ModelDomainObjectBuilder()
        # model_director      = MoelDomainObjectDirector()
        # model_domain_object = model_director.construct(builder= model_builder)
        filtering           = Filtering(model_domain_object)
        self.domain_object.set_filteirng(filtering)

    def get_domain_object(self) -> FilteringModelDomainObject:
        return self.domain_object

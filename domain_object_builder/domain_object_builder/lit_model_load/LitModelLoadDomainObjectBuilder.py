from omegaconf import DictConfig
from .LitModelLoadDomainObject import LitModelLoadDomainObject
# from cdsvae.domain.model import AbstractLitModel, LitModelFactory, AbstractModel
from domain_object_builder.lit_model import LitModelDomainObjectBuilder
from domain_object_director.lit_model import LitMoelDomainObjectDirector
from domain_object_builder.model import ModelDomainObject


class LitModelLoadDomainObjectBuilder:
    def __init__(self):
        self.domain_object = LitModelLoadDomainObject()

    def buid_lit_model_domain_object(self, model: AbstractModel, config_model: DictConfig):
        builder  = LitModelDomainObjectBuilder(model)
        director = LitMoelDomainObjectDirector()
        return director.construct(builder, model_class=config_model.model_class)

    def build_lit_model(self, model: AbstractModel, config_model : DictConfig = None):
        lit_model_domain_object = self.buid_lit_model_domain_object(model, config_model)
        lit_model = LitModelFactory.create(name=config_model.model_class, domain_object=lit_model_domain_object)
        self.domain_object.set_lit_model(lit_model)

    def build_lit_model_eval(self, model: AbstractModel, config_model : DictConfig = None):
        lit_model_domain_object = self.buid_lit_model_domain_object(model, config_model)
        # ---
        lit_model = LitModelFactory.load_from_checkpoint(
            name            = config_model.model_class,
            domain_object   = lit_model_domain_object,
            checkpoint_path = config_model.reload.path,
        )
        # ---
        lit_model.freeze()
        lit_model.eval().cuda()
        # ---
        self.domain_object.set_lit_model(lit_model)


    def get_lit_model(self) -> AbstractLitModel:
        return self.domain_object.lit_model

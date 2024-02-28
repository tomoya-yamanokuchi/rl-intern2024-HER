from omegaconf import DictConfig
from .LitEnsembleLoadDomainObject import LitEnsembleLoadDomainObject
from cdsvae.domain.model import AbstractLitEnsemble
from domain_object_builder.lit_ensemble import LitEnsembleDomainObjectBuilder
from domain_object_director.lit_ensemble import LitEnsembleDomainObjectDirector
from cdsvae.domain.model import LitModelFactory, AbstractEnsemble


class LitEnsembleLoadDomainObjectBuilder:
    def __init__(self):
        self.domain_object = LitEnsembleLoadDomainObject()

    def build_lit_ensemble_domain_object(self, model: AbstractEnsemble):
        builder  = LitEnsembleDomainObjectBuilder(model)
        director = LitEnsembleDomainObjectDirector()
        return director.construct(builder)

    def build_lit_ensemble(self, model: AbstractEnsemble, config_model : DictConfig = None):
        lit_model_domain_object = self.build_lit_ensemble_domain_object(model)
        lit_model = LitModelFactory.create(name=config_model.model_class, domain_object=lit_model_domain_object)
        self.domain_object.set_lit_model(lit_model)

    def build_lit_ensemble_eval(self, model: AbstractEnsemble, config_model : DictConfig = None):
        lit_model_domain_object = self.build_lit_ensemble_domain_object(model)
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


    def get_lit_model(self) -> AbstractLitEnsemble:
        return self.domain_object.lit_model

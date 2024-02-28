from domain_object_builder.lit_ensemble_load import LitEnsembleLoadDomainObjectBuilder, LitEnsembleLoadDomainObject
from ..AbstractDomainObjectDirector import AbstractDomainObjectDirector
from cdsvae.domain.model import AbstractEnsemble
from omegaconf import DictConfig


class LitEnsembleEvalLoadDomainObjectDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(
            builder             : LitEnsembleLoadDomainObjectBuilder,
            model               : AbstractEnsemble,
            config_model        : DictConfig,
        ) -> LitEnsembleLoadDomainObjectBuilder:

        builder.build_lit_ensemble_eval(model, config_model)

        return builder.get_lit_model()

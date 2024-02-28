from domain_object_builder.lit_ensemble import LitEnsembleDomainObject, LitEnsembleDomainObjectBuilder
from ..AbstractDomainObjectDirector import AbstractDomainObjectDirector
from omegaconf import DictConfig


class LitEnsembleDomainObjectDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(
            builder             : LitEnsembleDomainObjectBuilder,
        ) -> LitEnsembleDomainObject:

        # ---- config ----
        builder.build_config_loader()
        builder.build_config()
        # ---- model ----
        builder.build_training_loss()
        builder.build_validation_loss()
        # ---
        return builder.get_domain_object()

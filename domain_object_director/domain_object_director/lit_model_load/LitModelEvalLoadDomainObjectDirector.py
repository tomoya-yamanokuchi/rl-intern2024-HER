from domain_object_builder.lit_model_load import LitModelLoadDomainObject, LitModelLoadDomainObjectBuilder
from ..AbstractDomainObjectDirector import AbstractDomainObjectDirector
from omegaconf import DictConfig


class LitModelEvalLoadDomainObjectDirector(AbstractDomainObjectDirector):
    from cdsvae.domain.model import AbstractModel
    @staticmethod
    def construct(
            builder : LitModelLoadDomainObjectBuilder,
            model   : AbstractModel,
            config  : DictConfig,
        ) -> LitModelLoadDomainObject:
        # ----
        builder.build_lit_model_eval(model, config)
        # ----
        return builder.get_lit_model()

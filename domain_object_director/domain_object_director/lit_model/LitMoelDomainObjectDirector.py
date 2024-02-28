from domain_object_builder.lit_model import LitModelDomainObject, LitModelDomainObjectBuilder
from ..AbstractDomainObjectDirector import AbstractDomainObjectDirector
from domain_object_builder.model import ModelDomainObject


class LitMoelDomainObjectDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder : LitModelDomainObjectBuilder, model_class: str) -> LitModelDomainObject:
        # ---- config ----
        builder.build_config_loader()
        builder.build_config()
        # ---- logger ----
        builder.build_image_logger()
        # ---- loss ----
        builder.build_weight()
        builder.build_contrastive_mi_c()
        builder.build_contrastive_mi_m()
        builder.build_mutual_information()
        builder.build_training_loss(model_class)
        builder.build_validation_loss(model_class)
        # ---
        return builder.get_domain_object()

from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector
from omegaconf import DictConfig


class EnsembleTrainingDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, model_dir: str) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_ensemble()  # replaceable for ensemble : builder.build_config_ensemble()
        # ---- after config ----
        builder.build_ensemble_adapter(model_dir)
        builder.build_ensemble()           # replaceable for ensemble : builder.build_ensemble()
        builder.build_lit_ensemble(model_dir)
        builder.build_ensemble_datamodule(
            config       = builder.domain_object.config_ensemble,
            model_dir    = model_dir,
        )
        builder.build_training_modules(
            # replaceable for ensemble : builder.domain_object._config_ensemble
            config_model = builder.domain_object.config_ensemble,
            version      = "ensemble"
        )
        # ---
        return builder.get_domain_object()

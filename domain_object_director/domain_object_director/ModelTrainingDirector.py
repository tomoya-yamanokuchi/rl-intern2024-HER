from domain_object_builder import DomainObject, DomainObjectBuilder
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector


class ModelTrainingDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, env_name: str) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_model()        # replaceable for ensemble : builder.build_config_ensemble()
        builder.build_config_env(env_name)
        # ---- after config ----
        builder.build_task_space(env_name, mode="torch")
        builder.build_trajectory_evaluator()
        builder.build_adapter(env_name)
        builder.build_model_domain_object()
        builder.build_model()
        builder.build_lit_model_train()           # replaceable for ensemble : builder.build_ensemble()
        builder.build_model_datamodule(builder.domain_object.config_model.datamodule)
        builder.build_training_modules(
            # replaceable for ensemble : builder.domain_object._config_ensemble
            config_model = builder.domain_object.config_model
        )
        # ---
        return builder.get_domain_object()

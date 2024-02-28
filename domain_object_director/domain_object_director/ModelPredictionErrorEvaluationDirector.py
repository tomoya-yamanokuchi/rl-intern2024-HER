from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector
from omegaconf import DictConfig

class ModelPredictionErrorEvaluationDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, env_name: str, metadata: ModelSetParamsDict, config_datamodule) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_model()        # replaceable for ensemble : builder.build_config_ensemble()
        builder.build_config_env(env_name)
        builder.build_test_config_datamodule(config_datamodule)
        # ---- after config ----
        builder.build_task_space(env_name, mode="torch")
        builder.build_trajectory_evaluator()
        builder.build_adapter(env_name)
        builder.build_model_domain_object(config=metadata["config_model"])
        builder.build_model()
        builder.build_lit_model_eval(config=metadata["config_model"])
        # --- prediction ---
        builder.build_prediction_model()
        builder.build_prediction_manager()
        # --- eval specific ---
        builder.build_model_datamodule(config_datamodule=builder.domain_object.config_datamodule)
        # --
        builder.build_model_prediction_planning()
        # ---
        return builder.get_domain_object()

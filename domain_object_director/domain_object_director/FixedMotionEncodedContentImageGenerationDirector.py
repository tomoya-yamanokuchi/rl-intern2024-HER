from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector


class FixedMotionEncodedContentImageGenerationDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, env_name: str, metadata: ModelSetParamsDict) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_model()        # replaceable for ensemble : builder.build_config_ensemble()
        builder.build_config_env(env_name)
        # ---- after config ----
        builder.build_task_space(env_name, mode="torch")
        builder.build_trajectory_evaluator()
        builder.build_adapter(env_name)
        builder.build_model_domain_object(config=metadata["config_model"])
        builder.build_model()
        builder.build_model_dir(model_dir=metadata["model_dir"])
        builder.build_lit_model_eval(config=metadata["config_model"])
        builder.build_filtering_model()
        builder.build_filtering_manager()
        # --- eval specific ---
        builder.build_model_datamodule(config_datamodule=metadata["config_model"].datamodule)
        builder.build_image_viewer()
        # ---
        return builder.get_domain_object()

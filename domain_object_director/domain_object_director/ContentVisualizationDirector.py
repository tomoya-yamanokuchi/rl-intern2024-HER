from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector
from omegaconf import DictConfig


class ContentVisualizationDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(
            builder          : DomainObjectBuilder,
            env_name         : str,
            metadata         : ModelSetParamsDict,
            config_datamodule: DictConfig,
        ) -> DomainObject:
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
        builder.build_lit_model_eval(config=metadata["config_model"])
        # --- eval specific ---
        builder.build_model_datamodule(config_datamodule)
        builder.build_tsne_visualizer()
        builder.build_scatter_visualizer()
        # ---
        return builder.get_domain_object()

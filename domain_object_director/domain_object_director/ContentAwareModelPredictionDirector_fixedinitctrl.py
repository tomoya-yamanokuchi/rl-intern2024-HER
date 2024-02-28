from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector
from omegaconf import OmegaConf


class ContentAwareModelPredictionDirector_fixedinitctrl(AbstractDomainObjectDirector):
    @staticmethod
    def construct(
            builder          : DomainObjectBuilder,
            env_name         : str,
            metadata         : ModelSetParamsDict,
            config_datamodule: OmegaConf,
        ) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_model(config_model=metadata["config_model"])        # replaceable for ensemble : builder.build_config_ensemble()
        builder.build_config_env(env_name)
        # ---- after config ----
        builder.build_task_space(env_name, mode="torch")
        builder.build_trajectory_evaluator()
        builder.build_adapter(env_name)
        builder.build_model_domain_object()

        # ----- MPC -----
        builder.build_model()
        builder.build_lit_model_eval(config=metadata["config_model"])
        builder.build_filtering_model()
        builder.build_prediction_model()
        builder.build_filtering_manager()
        builder.build_prediction_manager()
        # ----
        builder.build_model_datamodule(config_datamodule)
        builder.build_model_planning_with_abs_ctrl_fixed_init_ctrl()
        # ---
        builder.build_image_logger()
        # ---
        return builder.get_domain_object()

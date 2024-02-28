from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector
from omegaconf import DictConfig


class ModelTrainingDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, env_name: str, dataset_dir: str) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_model()
        builder.build_config_env(env_name)
        # ---- after config ----
        builder.build_task_space(env_name, mode="torch")
        builder.build_adapter(env_name, dataset_dir)
        builder.build_datamodule()
        builder.build_model_domain_object()
        builder.build_lit_model()
        builder.build_tb_logger()
        builder.build_trainer()
        # ---
        return builder.get_domain_object()

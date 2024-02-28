from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector


class LoadRandomCtrlDatasetDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, env_name: str, dataset_dir : str) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_model()
        builder.build_config_env(env_name)
        # ---- after config ----
        builder.build_env_data_container()
        builder.build_shelve_repository(save_dir=None, read_only=True)
        builder.build_env_data_stack()
        builder.build_env_data_stack_info_writer()
        builder.build_dataset_visualizer(dataset_dir)
        return builder.get_domain_object()
